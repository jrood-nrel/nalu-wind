// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef StringTimeCoordTemperatureAuxFunction_h
#define StringTimeCoordTemperatureAuxFunction_h

#include "AuxFunction.h"
#include "StringTimeCoordFunction.h"

#include <vector>

namespace sierra::nalu {

class StringTimeCoordTemperatureAuxFunction : public AuxFunction
{
public:
  StringTimeCoordTemperatureAuxFunction(std::string fcn);

  using AuxFunction::do_evaluate;
  void do_evaluate(
    const double* coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double* fieldPtr,
    const unsigned fieldSize,
    const unsigned beginPos,
    const unsigned endPos) const;

private:
  const StringTimeCoordFunction f_;
};

} // namespace sierra::nalu

#endif
