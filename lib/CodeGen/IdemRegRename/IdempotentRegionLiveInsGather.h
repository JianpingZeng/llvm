#ifndef IDEMPOTENT_REGION_LIVEINS_GATHER_H
#define IDEMPOTENT_REGION_LIVEINS_GATHER_H

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/BitVector.h>
#include <map>
#include <set>
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {

class LiveInsGather {
private:
  const MachineFunction &mf;
  typedef std::set<unsigned> RegSet;

  const TargetInstrInfo *tii;
  const TargetRegisterInfo *tri;
  // Register liveness at idempotent region boundaries.
  std::map<const MachineInstr *, RegSet> idemLiveInMap;
  std::map<const MachineBasicBlock *, RegSet> liveInMBBMap;
  std::map<const MachineBasicBlock *, RegSet> liveOutMBBMap;
  std::map<const MachineBasicBlock *, RegSet> liveGens;
  std::map<const MachineBasicBlock *, RegSet> liveKills;

public:
  LiveInsGather(const MachineFunction &MF) : mf(MF), tii(MF.getTarget().getInstrInfo()),
                                             tri(MF.getTarget().getRegisterInfo()),
                                             idemLiveInMap(), liveInMBBMap(),
                                             liveOutMBBMap(),
                                             liveGens(), liveKills() { }

  void run();

  RegSet getIdemLiveIns(MachineInstr *mi) {
    assert(mi && tii->isIdemBoundary(mi));
    assert(idemLiveInMap.count(mi));
    return idemLiveInMap[mi];
  }

private:
  void computeIdemLiveIns(const MachineInstr *mi);
  void printLiveRegisters(RegSet &regs, bool liveInOrLiveOut = true);
};
}

#endif