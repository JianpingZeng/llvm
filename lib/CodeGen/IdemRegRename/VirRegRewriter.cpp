//===----- VirRegRewriter.cpp - Register Rewriter for register renaming ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------------===//

#include "VirRegRewriter.h"

using namespace llvm;

void VirRegRewriter::rewrite(std::vector<LiveIntervalIdem *> &handled,
                             std::map<LiveIntervalIdem*, unsigned> &interval2AssignedRegMap) {
  if (handled.empty()) return;

  for (LiveIntervalIdem *it : handled) {
    if (TargetRegisterInfo::isPhysicalRegister(it->reg))
      continue;

    assert(interval2AssignedRegMap.count(it) && "no free register");
    unsigned phyReg = interval2AssignedRegMap[it];
    for (const UsePoint &up : it->usePoints) {
      up.mo->setReg(phyReg);
    }
  }
}