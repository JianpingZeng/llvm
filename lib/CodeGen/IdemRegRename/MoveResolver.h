//===------- MoveResolver.h - Move resolver for register renaming ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MOVERESOLVER_H
#define LLVM_MOVERESOLVER_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "LiveIntervalAnalysisIdem.h"
#include "IdemRegisterRenaming2.h"
#include <vector>
#include <utility>

namespace llvm {
  class IdemRegisterRenamer;

  class MoveResolver {
  private:
    MachineBasicBlock* insertedMBB;
    MachineInstr *insertedPos;
    std::vector<std::pair<LiveIntervalIdem*, LiveIntervalIdem*>> mappings;
    int *registerBlocked;
    bool multipleReadsAllowed;
    const TargetRegisterInfo *tri;
    const TargetInstrInfo *tii;
    unsigned numRegs;
    IdemRegisterRenamer *renamer;

  private:
    void insertMove(LiveIntervalIdem *srcIt, LiveIntervalIdem *dstIt);
    void blockRegisters(LiveIntervalIdem *it);
    void unblockRegisters(LiveIntervalIdem *it);
    void setRegisterBlocked(unsigned reg, int direction);
    int getRegisterBlocked(unsigned reg);
    bool isSafeToProcessMove(LiveIntervalIdem *srcIt, LiveIntervalIdem *dstIt);

  public:
    MoveResolver(const TargetRegisterInfo *_tri,
                 const TargetInstrInfo *_tii,
                 unsigned _numRegs,
                 IdemRegisterRenamer *_renamer) :
        insertedMBB(0),
        insertedPos(0),
        mappings(),
        registerBlocked(0),
        multipleReadsAllowed(false),
        renamer(_renamer) {
      tri = _tri;
      tii = _tii;
      numRegs = _numRegs;
      registerBlocked = new int[numRegs];
    }

    void clear() {
      insertedMBB = nullptr;
      insertedPos = nullptr;
      mappings.clear();
      delete[] registerBlocked;
      multipleReadsAllowed = false;
      tri = nullptr;
      tii = nullptr;
      numRegs = 0;
    }

    void resolveMapping();
    void insertMoveInstr(MachineInstr *pos);
    void addMapping(LiveIntervalIdem *srcIt, LiveIntervalIdem *dstIt);
  };
}
#endif //LLVM_MOVERESOLVER_H
