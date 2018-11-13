//===----- VirRegRewriter.h - Register Rewriter for register renaming ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------------===//

#ifndef LLVM_VIRREGREWRITER_H
#define LLVM_VIRREGREWRITER_H

#include <vector>
#include "LiveIntervalAnalysisIdem.h"

namespace llvm {
class VirRegRewriter {
public:
  void rewrite(std::vector<LiveIntervalIdem *> &handled,
               std::map<LiveIntervalIdem *, unsigned> &interval2AssignedRegMap);
};

}

#endif //LLVM_VIRREGREWRITER_H
