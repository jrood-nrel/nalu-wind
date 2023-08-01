#ifdef NALU_USES_NALU_HYPRE

#include "HypreDirectSolver.h"
#include "XSDKHypreInterface.h"
#include "NaluEnv.h"

namespace sierra {
namespace nalu {

namespace {
// This anonymous namespace contains wrapper methods to NALU_HYPRE solver creation
// methods. It hides around the fact that some solvers require an MPI
// communicator while others do not. This allows HypreDirectSolver::CreateSolver
// methods to assign pointers using the same function signature.
//
// Note that this section has been modeled after xSDK Trilinos package. See
// <https://github.com/trilinos/xSDKTrilinos/blob/master/nalu_hypre/src/Ifpack2_Hypre.hpp>
// for more details

HypreIntType
Hypre_BoomerAMGCreate(MPI_Comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_BoomerAMGCreate(solver);
}

HypreIntType
Hypre_ParaSailsCreate(MPI_Comm comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_ParaSailsCreate(comm, solver);
}

HypreIntType
Hypre_EuclidCreate(MPI_Comm comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_EuclidCreate(comm, solver);
}

HypreIntType
Hypre_AMSCreate(MPI_Comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_AMSCreate(solver);
}

HypreIntType
Hypre_ParCSRHybridCreate(MPI_Comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_ParCSRHybridCreate(solver);
}

HypreIntType
Hypre_ParCSRPCGCreate(MPI_Comm comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_ParCSRPCGCreate(comm, solver);
}

HypreIntType
Hypre_ParCSRGMRESCreate(MPI_Comm comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_ParCSRGMRESCreate(comm, solver);
}

HypreIntType
Hypre_ParCSRCOGMRESCreate(MPI_Comm comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_ParCSRCOGMRESCreate(comm, solver);
}

HypreIntType
Hypre_ParCSRFlexGMRESCreate(MPI_Comm comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_ParCSRFlexGMRESCreate(comm, solver);
}

HypreIntType
Hypre_ParCSRLGMRESCreate(MPI_Comm comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_ParCSRLGMRESCreate(comm, solver);
}

HypreIntType
Hypre_ParCSRBiCGSTABCreate(MPI_Comm comm, NALU_HYPRE_Solver* solver)
{
  return NALU_HYPRE_ParCSRBiCGSTABCreate(comm, solver);
}
} // namespace

HypreDirectSolver::HypreDirectSolver(
  std::string name,
  HypreLinearSolverConfig* config,
  LinearSolvers* linearSolvers)
  : LinearSolver(name, linearSolvers, config)
{
}

HypreDirectSolver::~HypreDirectSolver() { destroyLinearSolver(); }

int
HypreDirectSolver::solve(
  int& numIterations, double& finalResidualNorm, bool isFinalOuterIter)
{
  // Initialize the solver on first entry
  double time = -NaluEnv::self().nalu_time();
  if (initializeSolver_)
    initSolver();
  time += NaluEnv::self().nalu_time();
  timerPrecond_ = time;

  numIterations = 0;
  finalResidualNorm = 0.0;

  // Can use the return value from solverSolvePtr_. However, Hypre seems to
  // return a non-zero value and that causes spurious error message output in
  // Nalu.
  int status = 0;

  if (isFinalOuterIter)
    solverSetTolPtr_(solver_, config_->finalTolerance());
  else
    solverSetTolPtr_(solver_, config_->tolerance());

  // Solve the system Ax = b
  solverSolvePtr_(solver_, parMat_, parRhs_, parSln_);

  // Extract linear num. iterations and linear residual. Unlike the TPetra
  // interface, Hypre returns the relative residual norm and not the final
  // absolute residual.
  HypreIntType numIters;
  solverNumItersPtr_(solver_, &numIters);
  solverFinalResidualNormPtr_(solver_, &finalResidualNorm);
  numIterations = numIters;

  return status;
}

void
HypreDirectSolver::set_initialize_solver_flag()
{
  /* used for tracking how often to reinit the solver/preconditioner */
  internalIterCounter_++;

  if (!config_->recomputePreconditioner() || config_->reusePreconditioner())
    initializeSolver_ = false;
  else {
    if (internalIterCounter_ % config_->recomputePrecondFrequency() == 0)
      initializeSolver_ = true;
    else
      initializeSolver_ = false;
  }
}

void
HypreDirectSolver::destroyLinearSolver()
{
  if (isSolverSetup_)
    solverDestroyPtr_(solver_);
  isSolverSetup_ = false;

  if (isPrecondSetup_)
    precondDestroyPtr_(precond_);
  isPrecondSetup_ = false;
}

void
HypreDirectSolver::initSolver()
{
  namespace Hypre = Ifpack2::Hypre;

  auto plist = config_->paramsPrecond();

  solverType_ = plist->get("Solver", Hypre::GMRES);
  usePrecond_ = plist->get("SetPreconditioner", false);
  if (usePrecond_)
    precondType_ = plist->get("Preconditioner", Hypre::BoomerAMG);

  Hypre::Hypre_Chooser chooser =
    plist->get("SolveOrPrecondition", Hypre::Solver);
  if (chooser != Hypre::Solver)
    throw std::runtime_error("HypreDirectSolver::initParameters: Invalid "
                             "option provided for Hypre Solver");

  // Everything checks out... create the solver and preconditioner
  createSolver();
  if (usePrecond_)
    createPrecond();

  // Apply user configuration parameters to solver and precondtioner
  int numFuncs = plist->get("NumFunctions", 0);
  if (numFuncs > 0) {
    Teuchos::RCP<Ifpack2::FunctionParameter>* params =
      plist->get<Teuchos::RCP<Ifpack2::FunctionParameter>*>("Functions");

    for (int i = 0; i < numFuncs; i++) {
      params[i]->CallFunction(solver_, precond_);
    }
  }

  if (usePrecond_)
    solverPrecondPtr_(solver_, precondSolvePtr_, precondSetupPtr_, precond_);

  setupSolver();

  /* solver is setup so set this flag to false */
  initializeSolver_ = false;
}

void
HypreDirectSolver::setupSolver()
{
  // We are always using NALU_HYPRE solver
  solverSetupPtr_(solver_, parMat_, parRhs_, parSln_);
}

void
HypreDirectSolver::createSolver()
{
  namespace Hypre = Ifpack2::Hypre;

  if (isSolverSetup_) {
    solverDestroyPtr_(solver_);
    isSolverSetup_ = false;
  }

  switch (solverType_) {
  case Hypre::BoomerAMG:
    solverCreatePtr_ = &Hypre_BoomerAMGCreate;
    solverDestroyPtr_ = &NALU_HYPRE_BoomerAMGDestroy;
    solverSetupPtr_ = &NALU_HYPRE_BoomerAMGSetup;
    solverPrecondPtr_ = nullptr;
    solverSolvePtr_ = &NALU_HYPRE_BoomerAMGSolve;
    solverSetTolPtr_ = &NALU_HYPRE_BoomerAMGSetTol;
    solverNumItersPtr_ = &NALU_HYPRE_BoomerAMGGetNumIterations;
    solverFinalResidualNormPtr_ = &NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm;
    break;

  case Hypre::GMRES:
    solverCreatePtr_ = &Hypre_ParCSRGMRESCreate;
    solverDestroyPtr_ = &NALU_HYPRE_ParCSRGMRESDestroy;
    solverSetupPtr_ = &NALU_HYPRE_ParCSRGMRESSetup;
    solverPrecondPtr_ = &NALU_HYPRE_ParCSRGMRESSetPrecond;
    solverSolvePtr_ = &NALU_HYPRE_ParCSRGMRESSolve;
    solverSetTolPtr_ = &NALU_HYPRE_ParCSRGMRESSetTol;
    solverNumItersPtr_ = &NALU_HYPRE_GMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &NALU_HYPRE_GMRESGetFinalRelativeResidualNorm;
    break;

  case Hypre::COGMRES:
    solverCreatePtr_ = &Hypre_ParCSRCOGMRESCreate;
    solverDestroyPtr_ = &NALU_HYPRE_ParCSRCOGMRESDestroy;
    solverSetupPtr_ = &NALU_HYPRE_ParCSRCOGMRESSetup;
    solverPrecondPtr_ = &NALU_HYPRE_ParCSRCOGMRESSetPrecond;
    solverSolvePtr_ = &NALU_HYPRE_ParCSRCOGMRESSolve;
    solverSetTolPtr_ = &NALU_HYPRE_ParCSRCOGMRESSetTol;
    solverNumItersPtr_ = &NALU_HYPRE_COGMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &NALU_HYPRE_COGMRESGetFinalRelativeResidualNorm;
    break;

  case Hypre::FlexGMRES:
    solverCreatePtr_ = &Hypre_ParCSRFlexGMRESCreate;
    solverDestroyPtr_ = &NALU_HYPRE_ParCSRFlexGMRESDestroy;
    solverSetupPtr_ = &NALU_HYPRE_ParCSRFlexGMRESSetup;
    solverPrecondPtr_ = &NALU_HYPRE_ParCSRFlexGMRESSetPrecond;
    solverSolvePtr_ = &NALU_HYPRE_ParCSRFlexGMRESSolve;
    solverSetTolPtr_ = &NALU_HYPRE_ParCSRFlexGMRESSetTol;
    solverNumItersPtr_ = &NALU_HYPRE_FlexGMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm;
    break;

  case Hypre::LGMRES:
    solverCreatePtr_ = &Hypre_ParCSRLGMRESCreate;
    solverDestroyPtr_ = &NALU_HYPRE_ParCSRLGMRESDestroy;
    solverSetupPtr_ = &NALU_HYPRE_ParCSRLGMRESSetup;
    solverPrecondPtr_ = &NALU_HYPRE_ParCSRLGMRESSetPrecond;
    solverSolvePtr_ = &NALU_HYPRE_ParCSRLGMRESSolve;
    solverSetTolPtr_ = &NALU_HYPRE_ParCSRLGMRESSetTol;
    solverNumItersPtr_ = &NALU_HYPRE_LGMRESGetNumIterations;
    solverFinalResidualNormPtr_ = &NALU_HYPRE_LGMRESGetFinalRelativeResidualNorm;
    break;

  case Hypre::BiCGSTAB:
    solverCreatePtr_ = &Hypre_ParCSRBiCGSTABCreate;
    solverDestroyPtr_ = &NALU_HYPRE_ParCSRBiCGSTABDestroy;
    solverSetupPtr_ = &NALU_HYPRE_ParCSRBiCGSTABSetup;
    solverPrecondPtr_ = &NALU_HYPRE_ParCSRBiCGSTABSetPrecond;
    solverSolvePtr_ = &NALU_HYPRE_ParCSRBiCGSTABSolve;
    solverSetTolPtr_ = &NALU_HYPRE_ParCSRBiCGSTABSetTol;
    solverNumItersPtr_ = &NALU_HYPRE_BiCGSTABGetNumIterations;
    solverFinalResidualNormPtr_ = &NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm;
    break;

  case Hypre::AMS:
    solverCreatePtr_ = &Hypre_AMSCreate;
    solverDestroyPtr_ = &NALU_HYPRE_AMSDestroy;
    solverSetupPtr_ = &NALU_HYPRE_AMSSetup;
    solverPrecondPtr_ = nullptr;
    solverSolvePtr_ = &NALU_HYPRE_AMSSolve;
    solverSetTolPtr_ = &NALU_HYPRE_AMSSetTol;
    solverNumItersPtr_ = &NALU_HYPRE_AMSGetNumIterations;
    solverFinalResidualNormPtr_ = &NALU_HYPRE_AMSGetFinalRelativeResidualNorm;
    break;

  case Hypre::PCG:
    solverCreatePtr_ = &Hypre_ParCSRPCGCreate;
    solverDestroyPtr_ = &NALU_HYPRE_ParCSRPCGDestroy;
    solverSetupPtr_ = &NALU_HYPRE_ParCSRPCGSetup;
    solverPrecondPtr_ = &NALU_HYPRE_ParCSRPCGSetPrecond;
    solverSolvePtr_ = &NALU_HYPRE_ParCSRPCGSolve;
    solverSetTolPtr_ = &NALU_HYPRE_ParCSRPCGSetTol;
    solverNumItersPtr_ = &NALU_HYPRE_PCGGetNumIterations;
    solverFinalResidualNormPtr_ = &NALU_HYPRE_PCGGetFinalRelativeResidualNorm;
    break;

  case Hypre::Hybrid:
    solverCreatePtr_ = &Hypre_ParCSRHybridCreate;
    solverDestroyPtr_ = &NALU_HYPRE_ParCSRHybridDestroy;
    solverSetupPtr_ = &NALU_HYPRE_ParCSRHybridSetup;
    solverPrecondPtr_ = &NALU_HYPRE_ParCSRHybridSetPrecond;
    solverSolvePtr_ = &NALU_HYPRE_ParCSRHybridSolve;
    solverSetTolPtr_ = &NALU_HYPRE_ParCSRHybridSetTol;
    solverNumItersPtr_ = &NALU_HYPRE_ParCSRHybridGetNumIterations;
    solverFinalResidualNormPtr_ =
      &NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm;
    break;

  default:
    solverCreatePtr_ = nullptr;
    break;
  }

  if (solverCreatePtr_ == nullptr)
    throw std::runtime_error("Error initializing NALU_HYPRE Solver");

  solverCreatePtr_(comm_, &solver_);
  isSolverSetup_ = true;
}

void
HypreDirectSolver::createPrecond()
{
  namespace Hypre = Ifpack2::Hypre;

  if (isPrecondSetup_) {
    precondDestroyPtr_(precond_);
    isPrecondSetup_ = false;
  }

  switch (precondType_) {
  case Hypre::BoomerAMG:
    precondCreatePtr_ = &Hypre_BoomerAMGCreate;
    precondDestroyPtr_ = &NALU_HYPRE_BoomerAMGDestroy;
    precondSetupPtr_ = &NALU_HYPRE_BoomerAMGSetup;
    precondSolvePtr_ = &NALU_HYPRE_BoomerAMGSolve;
    break;

  case Hypre::Euclid:
    precondCreatePtr_ = &Hypre_EuclidCreate;
    precondDestroyPtr_ = &NALU_HYPRE_EuclidDestroy;
    precondSetupPtr_ = &NALU_HYPRE_EuclidSetup;
    precondSolvePtr_ = &NALU_HYPRE_EuclidSolve;
    break;

  case Hypre::ParaSails:
    precondCreatePtr_ = &Hypre_ParaSailsCreate;
    precondDestroyPtr_ = &NALU_HYPRE_ParaSailsDestroy;
    precondSetupPtr_ = &NALU_HYPRE_ParaSailsSetup;
    precondSolvePtr_ = &NALU_HYPRE_ParaSailsSolve;
    break;

  case Hypre::AMS:
    precondCreatePtr_ = &Hypre_AMSCreate;
    precondDestroyPtr_ = &NALU_HYPRE_AMSDestroy;
    precondSetupPtr_ = &NALU_HYPRE_AMSSetup;
    precondSolvePtr_ = &NALU_HYPRE_AMSSolve;
    break;

  default:
    precondCreatePtr_ = nullptr;
    break;
  }

  if (precondCreatePtr_ == nullptr)
    throw std::runtime_error("Error initializing NALU_HYPRE Preconditioner");

  precondCreatePtr_(comm_, &precond_);
  isPrecondSetup_ = true;
}

} // namespace nalu
} // namespace sierra

#endif
