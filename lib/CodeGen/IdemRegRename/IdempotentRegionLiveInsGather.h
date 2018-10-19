#ifndef IDEMPOTENT_REGION_LIVEINS_GATHER_H
#define IDEMPOTENT_REGION_LIVEINS_GATHER_H

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/BitVector.h>
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
  typedef DenseSet<unsigned> RegSet;

  RegSet regsLiveInButUnused;
  const TargetInstrInfo *tii;
  const TargetRegisterInfo *tri;
  // Register liveness at idempotent region boundaries.
  DenseMap<const MachineInstr*, RegSet> idemLiveInMap;
  DenseMap<const MachineBasicBlock*, RegSet> liveInMBBMap;
  DenseMap<const MachineBasicBlock*, RegSet> liveOutMBBMap;
  DenseMap<const MachineBasicBlock*, RegSet> liveGens;
  DenseMap<const MachineBasicBlock*, RegSet> liveKills;

public:
  LiveInsGather(const MachineFunction &MF) : mf(MF) {
    tii = MF.getTarget().getInstrInfo();
    tri = MF.getTarget().getRegisterInfo();
  }
  void run();

  RegSet getIdemLiveIns(MachineInstr *mi) {
    assert(mi && tii->isIdemBoundary(mi));
    assert(idemLiveInMap.count(mi));
    return idemLiveInMap[mi];
  }

private:
  void computeIdemLiveIns(const MachineInstr *mi);
};
}

#endif