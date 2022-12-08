import os
import argparse
from collections import deque
from copy import deepcopy
from inspect import stack

MemSize = 1000  # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.


class InsMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        with open(ioDir + "/imem.txt") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def readInstr(self, ReadAddress):
        # read instruction memory
        # return 32 bit hex val
        bin32 = 0
        for i in range(ReadAddress, ReadAddress + 4):
            bin32 = bin32 | int('0b' + self.IMem[i], 2)
            if i < ReadAddress + 3:
                bin32 <<= 8

        return '0x' + '0' * (8 - len(hex(bin32)[2:])) + hex(bin32)[2:]


class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(ioDir + "/dmem.txt") as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]
        self.DMem = self.DMem + (['00000000'] * (MemSize - len(self.DMem)))

    def readDataMem(self, ReadAddress):
        # read data memory
        # return 32 bit hex val
        bin32 = 0
        for i in range(ReadAddress, ReadAddress + 4):
            bin32 = bin32 | int('0b' + self.DMem[i], 2)
            if i < ReadAddress + 3:
                bin32 <<= 8

        return '0x' + '0' * (8 - len(hex(bin32)[2:])) + hex(bin32)[2:]

    def writeDataMem(self, Address, WriteData):
        # write data into byte addressable memory
        bitmask = 2 ** 8 - 1
        bin8_arr = []

        for _ in range(4):
            bin8_arr.append(WriteData & bitmask)
            WriteData >>= 8

        for i in range(4):
            self.DMem[Address + i] = '0' * (8 - len(bin(bin8_arr[-1])[2:])) + bin(bin8_arr[-1])[2:]
            bin8_arr.pop()

    def outputDataMem(self):
        resPath = self.ioDir + "\\" + self.id + "_DMEMResult.txt"
        with open(resPath, "w") as rp:
            rp.writelines([data + "\n" for data in self.DMem])


class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        self.Registers = [0x0 for i in range(32)]

    def readRF(self, Reg_addr):
        # Fill in
        return self.Registers[Reg_addr]

    def writeRF(self, Reg_addr, Wrt_reg_data):
        # Fill in
        self.Registers[Reg_addr] = Wrt_reg_data & ((1 << 32) - 1)

    def outputRF(self, cycle):
        op = ["-" * 70 + "\n", "State of RF after executing cycle:        " + str(cycle) + "\n"]
        op.extend(['0' * (32 - len(bin(val)[2:])) + bin(val)[2:] + "\n" for val in self.Registers])
        if (cycle == 0):
            perm = "w"
        else:
            perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)


class State(object):
    def __init__(self):
        self.STAGES = deque()
        self.IF = {"nop": False, "PC": 0, "taken": False, "bubble": False}
        self.ID = {"nop": False, "Instr": 0, "PC": 0}
        self.EX = {"nop": False, "Imm": 0, "rs1": 0, "rs2": 0, "rd": 0, "wrt_mem": 0, "alu_op": 0, "wrt_enable": 0,
                   "funct3": 0, "funct7": 0, "imm": 0, "forward_reg": 0, "forward_data": 0, "PC": 0}
        self.MEM = {"nop": False, "ALUresult": 0, "Store_data": 0, "wrt_reg_addr": 0, "rd_mem": 0,
                    "wrt_mem": 0, "wrt_enable": False, "read_enable": 0, "forward_reg": 0, "forward_data": 0}
        self.WB = {"nop": False, "wrt_data": 0, "Rs": 0, "Rt": 0, "wrt_reg_addr": 0, "wrt_enable": 0}


class Core(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem


def getRdResultsR(funct7, funct3, data_rs1, data_rs2):
    data_rd = 0
    # ADD
    if funct7 == 0b0000000 and funct3 == 0b000:
        data_rd = data_rs1 + data_rs2

    # SUB
    if funct7 == 0b0100000 and funct3 == 0b000:
        data_rd = data_rs1 - data_rs2

    # XOR
    if funct7 == 0b0000000 and funct3 == 0b100:
        data_rd = data_rs1 ^ data_rs2

    # OR
    if funct7 == 0b0000000 and funct3 == 0b110:
        data_rd = data_rs1 | data_rs2

    # AND
    if funct7 == 0b0000000 and funct3 == 0b111:
        data_rd = data_rs1 & data_rs2

    return data_rd


def getRdResultsI(funct3, data_rs1, data_imm):
    data_rd = 0
    # ADDI
    if funct3 == 0b000:
        data_rd = data_rs1 + twos_comp(val=data_imm, sign_bit=11)

    # XORI
    if funct3 == 0b100:
        data_rd = data_rs1 ^ twos_comp(val=data_imm, sign_bit=11)

    # ORI
    if funct3 == 0b110:
        data_rd = data_rs1 | twos_comp(val=data_imm, sign_bit=11)

    # ANDI
    if funct3 == 0b111:
        data_rd = data_rs1 & twos_comp(val=data_imm, sign_bit=11)

    return data_rd


def twos_comp(val, sign_bit):
    """compute the 2's complement of int value val"""

    if (val & (1 << sign_bit)) != 0:  # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << (sign_bit + 1))  # compute negative value
    return val  # return positive value as is


class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(ioDir + "\\SS_", imem, dmem)
        self.opFilePath = ioDir + "\\StateResult_SS.txt"

    def step(self):

        # Your implementation
        fetchedInstr = int(self.ext_imem.readInstr(self.state.IF["PC"]), 16)
        opcode = fetchedInstr & (2 ** 7 - 1)

        # R-type
        if opcode == 0b0110011:

            # get funct7
            funct7 = fetchedInstr >> 25
            # get funct3
            funct3 = (fetchedInstr >> 12) & ((1 << 3) - 1)
            # get rs2
            rs2 = (fetchedInstr >> 20) & ((1 << 5) - 1)
            # get rs1
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            # get rd
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            # get data in rs1
            data_rs1 = self.myRF.readRF(rs1)
            # get data in rs2
            data_rs2 = self.myRF.readRF(rs2)
            # get result data
            data_rd = getRdResultsR(funct7=funct7, funct3=funct3, data_rs1=data_rs1, data_rs2=data_rs2)
            # store all fetched and computed data
            self.myRF.writeRF(rd, data_rd)

        # I Type
        elif opcode == 0b0010011:

            # get immediate
            imm = fetchedInstr >> 20 & ((1 << 12) - 1)

            # get funct3
            funct3 = (fetchedInstr >> 12) & ((1 << 3) - 1)
            # get rs1
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            # get rd
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            # get data in rs1
            data_rs1 = self.myRF.readRF(rs1)
            # get result data
            data_rd = getRdResultsI(funct3=funct3, data_rs1=data_rs1, data_imm=imm)
            # store result data in rd register
            self.myRF.writeRF(rd, data_rd)

        # J Type
        elif opcode == 0b1101111:

            # get imm
            imm19_12 = (fetchedInstr >> 12) & ((1 << 8) - 1)
            imm11 = (fetchedInstr >> 20) & 1
            imm10_1 = (fetchedInstr >> 21) & ((1 << 10) - 1)
            imm20 = (fetchedInstr >> 31) & 1
            imm = (imm20 << 20) | (imm10_1 << 1) | (imm11 << 11) | (imm19_12 << 12)

            # get rd
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            self.myRF.writeRF(rd, self.state.IF["PC"] + 4)
            self.nextState.IF["PC"] = self.state.IF["PC"] + twos_comp(val=imm, sign_bit=20)
            self.state.IF["taken"] = True

        # B Type
        elif opcode == 0b1100011:

            # get imm
            imm11 = (fetchedInstr >> 7) & 1
            imm4_1 = (fetchedInstr >> 8) & ((1 << 4) - 1)
            imm10_5 = (fetchedInstr >> 25) & ((1 << 6) - 1)
            imm12 = (fetchedInstr >> 31) & 1
            imm = (imm11 << 11) | (imm4_1 << 1) | (imm10_5 << 5) | (imm12 << 12)

            # get rs2
            rs2 = (fetchedInstr >> 20) & ((1 << 5) - 1)
            # get rs1
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            # get funct3
            funct3 = (fetchedInstr >> 12) & ((1 << 3) - 1)

            # BEQ
            if funct3 == 0b000:
                data_rs1 = self.myRF.readRF(rs1)
                data_rs2 = self.myRF.readRF(rs2)
                if data_rs1 == data_rs2:
                    self.nextState.IF["PC"] = self.state.IF["PC"] + twos_comp(val=imm, sign_bit=12)
                    self.state.IF["taken"] = True

            # BNE
            else:
                data_rs1 = self.myRF.readRF(rs1)
                data_rs2 = self.myRF.readRF(rs2)
                if data_rs1 != data_rs2:
                    self.nextState.IF["PC"] = self.state.IF["PC"] + twos_comp(val=imm, sign_bit=12)
                    self.state.IF["taken"] = True

        # LW
        elif opcode == 0b0000011:

            # get imm
            imm = fetchedInstr >> 20
            # get rs1
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            # get rd
            rd = (fetchedInstr >> 7) & ((1 << 5) - 1)

            self.myRF.writeRF(Reg_addr=rd,
                              Wrt_reg_data=int(self.ext_dmem.readDataMem(
                                  ReadAddress=self.myRF.readRF(rs1) + twos_comp(val=imm, sign_bit=11)), 16))

        # SW
        elif opcode == 0b0100011:

            # get imm
            imm11_5 = fetchedInstr >> 25
            imm4_0 = (fetchedInstr >> 7) & ((1 << 5) - 1)
            imm = (imm11_5 << 5) | imm4_0

            # get funct3
            funct3 = fetchedInstr & (((1 << 3) - 1) << 12)
            # get rs1
            rs1 = (fetchedInstr >> 15) & ((1 << 5) - 1)
            # get rd
            rs2 = (fetchedInstr >> 20) & ((1 << 5) - 1)

            self.ext_dmem.writeDataMem(Address=(rs1 + twos_comp(val=imm, sign_bit=11)) & ((1 << 32) - 1),
                                       WriteData=self.myRF.readRF(rs2))

        # HALT
        else:
            self.state.IF["nop"] = True

        self.halted = False
        if self.state.IF["nop"]:
            self.halted = True

        if not self.state.IF["taken"] and self.state.IF["PC"] + 4 < len(self.ext_imem.IMem):
            self.nextState.IF["PC"] = self.state.IF["PC"] + 4
        else:
            self.state.IF["taken"] = False

        self.myRF.outputRF(self.cycle)  # dump RF
        self.printState(self.nextState, self.cycle)  # print states after executing cycle 0, cycle 1, cycle 2 ...

        self.state = self.nextState  # The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-" * 70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")

        if (cycle == 0):
            perm = "w"
        else:
            perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(ioDir + "\\FS_", imem, dmem)
        self.opFilePath = ioDir + "\\StateResult_FS.txt"
        self.stages = deque()

    def step(self):

        # Your implementation
        self.state.IF["bubble"] = False
        for i in range(len(self.state.STAGES)):

            stage = self.state.STAGES[i]
            # --------------------- WB stage ---------------------
            if stage == 'WB':
                if self.state.MEM["nop"]:
                    self.nextState.WB["nop"] = True
                    self.nextState.MEM["nop"] = True
                    self.nextState.EX["nop"] = True
                    self.nextState.ID["nop"] = True
                    self.nextState.IF["nop"] = True

                elif not self.state.IF["bubble"]:

                    if self.state.WB["wrt_enable"]:

                        wrt_reg_addr = self.state.WB["wrt_reg_addr"]
                        wrt_data = self.state.WB["wrt_data"]

                        self.myRF.writeRF(Reg_addr=wrt_reg_addr, Wrt_reg_data=wrt_data)

                    self.state.IF["bubble"] = False

            # --------------------- MEM stage --------------------
            if stage == 'MEM':

                self.nextState.STAGES.append('WB')

                if self.state.EX["nop"]:

                    self.nextState.MEM["nop"] = True
                    self.nextState.EX["nop"] = True
                    self.nextState.ID["nop"] = True
                    self.nextState.IF["nop"] = True

                elif not self.state.IF["bubble"]:

                    if self.state.MEM["wrt_enable"]:
                        wrt_mem = self.state.MEM["wrt_mem"]
                        store_data = self.state.MEM["Store_data"]

                        self.ext_dmem.writeDataMem(Address=wrt_mem, WriteData=store_data)

                    elif self.state.MEM["read_enable"]:

                        rd_mem = self.state.MEM["rd_mem"]
                        # store data to be written in WB stage
                        self.nextState.WB["wrt_data"] = int(self.ext_dmem.readDataMem(ReadAddress=rd_mem), 16)
                        # store register address to write data in WB stage
                        self.nextState.WB["wrt_reg_addr"] = self.state.MEM["wrt_reg_addr"]
                        # enable writing data into the register
                        self.nextState.WB["wrt_enable"] = True
                        self.nextState.MEM["forward_reg"] = self.state.MEM["wrt_reg_addr"]
                        self.nextState.MEM["forward_data"] = int(self.ext_dmem.readDataMem(ReadAddress=rd_mem), 16)

                    else:

                        self.nextState.WB["wrt_data"] = self.state.MEM["ALUresult"]
                        self.nextState.WB["wrt_enable"] = True
                        self.nextState.MEM["forward_reg"] = self.state.MEM["wrt_reg_addr"]
                        self.nextState.WB["wrt_reg_addr"] = self.state.MEM["wrt_reg_addr"]

            # --------------------- EX stage ---------------------
            if stage == 'EX':
                self.nextState.STAGES.append('MEM')
                if self.state.ID["nop"]:

                    self.nextState.EX["nop"] = True
                    self.nextState.ID["nop"] = True
                    self.nextState.IF["nop"] = True

                elif not self.state.IF["bubble"]:

                    # R-type
                    if self.state.EX["alu_op"] == 0b0110011:
                        # get all operation-required segments
                        rd = self.state.EX["rd"]
                        funct3 = self.state.EX["funct3"]
                        funct7 = self.state.EX["funct7"]
                        rs1 = self.state.EX["rs1"]
                        rs2 = self.state.EX["rs2"]
                        rs1_value = self.myRF.readRF(rs1)
                        rs2_value = self.myRF.readRF(rs2)
                        forward_reg_EX = self.state.EX["forward_reg"]
                        forward_data_EX = self.state.EX["forward_data"]
                        forward_reg_MEM = self.state.MEM["forward_reg"]
                        forward_data_MEM = self.state.MEM["forward_data"]

                        if self.state.MEM["read_enable"] and (forward_reg_MEM == rs1 or forward_reg_MEM == rs2):
                            self.nextState.STAGES[-1] = 'EX'
                            self.nextState.EX = deepcopy(self.state.EX)

                        # forwarding if registers rs1 are not updated from the last R/I type instruction
                        if rs1 == forward_reg_EX:
                            rs1_value = forward_data_EX

                        # forwarding if registers rs2 are not updated from the last R/I type instruction
                        if rs2 == forward_reg_EX:
                            rs2_value = forward_data_EX

                        # forwarding if registers rs1 are not updated from the last lw instruction
                        if self.state.MEM["read_enable"] and rs1 == forward_reg_MEM:
                            rs1_value = forward_data_MEM

                        # forwarding if registers rs2 are not updated from the last lw instruction
                        if self.state.MEM["read_enable"] and rs2 == forward_reg_MEM:
                            rs2_value = forward_data_MEM

                        data_rd = getRdResultsR(funct3=funct3, funct7=funct7, data_rs1=rs1_value, data_rs2=rs2_value)
                        # record rd register address in the next state for MEM stage
                        self.nextState.MEM["wrt_reg_addr"] = rd
                        # update forward_reg for checking if the next instruction uses the rd register
                        self.nextState.EX["forward_reg"] = rd
                        # record alu results (i.e., instruction output) in the next state for MEM stage
                        self.nextState.MEM["ALUresult"] = data_rd
                        # update forward_data for future data forwarding
                        self.nextState.EX["forward_data"] = data_rd

                    # I Type
                    elif self.state.EX["alu_op"] == 0b0010011:
                        rs1 = self.state.EX["rs1"]
                        rd = self.state.EX["rd"]
                        funct3 = self.state.EX["funct3"]
                        imm = self.state.EX["imm"]
                        rs1_value = self.myRF.readRF(rs1)
                        forward_reg = self.state.EX["forward_reg"]
                        forward_data = self.state.EX["forward_data"]
                        forward_reg_MEM = self.state.MEM["forward_reg"]
                        forward_data_MEM = self.state.MEM["forward_data"]

                        if self.state.MEM["read_enable"] and forward_reg_MEM == rs1:
                            self.nextState.STAGES[-1] = 'EX'
                            self.nextState.EX = deepcopy(self.state.EX)

                        # forwarding if registers rs1 are not updated from the last instruction
                        if rs1 == forward_reg:
                            rs1_value = forward_data

                        # forwarding if registers rs1 are not updated from the last lw instruction
                        if self.state.MEM["read_enable"] and rs1 == forward_reg_MEM:
                            rs1_value = forward_data_MEM

                        data_rd = getRdResultsI(funct3=funct3, data_rs1=rs1_value, data_imm=imm)
                        # record rd register address in the next state for MEM stage
                        self.nextState.MEM["wrt_reg_addr"] = rd
                        # update forward_reg for checking if the next instruction uses the rd register
                        self.nextState.EX["forward_reg"] = rd
                        # record alu results (i.e., instruction output) in the next state for MEM stage
                        self.nextState.MEM["ALUresult"] = data_rd
                        # update forward_data for future data forwarding
                        self.nextState.EX["forward_data"] = data_rd

                    # J Type
                    elif self.state.EX["alu_op"] == 0b1101111:
                        rd = self.state.EX["rd"]

                        data_rd = self.state.IF["PC"] + 4

                        # record rd register address in the next state for MEM stage
                        self.nextState.MEM["wrt_reg_addr"] = rd
                        # update forward_reg for checking if the next instruction uses the rd register
                        self.nextState.EX["forward_reg"] = rd
                        # record alu results (i.e., instruction output) in the next state for MEM stage
                        self.nextState.MEM["ALUresult"] = data_rd
                        # update forward_data for future data forwarding
                        self.nextState.EX["forward_data"] = data_rd

                    # B Type
                    elif self.state.EX["alu_op"] == 0b1100011:

                        # self.state.IF["bubble"] = True

                        rs2 = self.state.EX["rs2"]
                        rs1 = self.state.EX["rs1"]
                        data_rs1 = self.myRF.readRF(rs1)
                        data_rs2 = self.myRF.readRF(rs2)
                        forward_reg = self.state.EX["forward_reg"]
                        forward_data = self.state.EX["forward_data"]
                        funct3 = self.state.EX["funct3"]

                        if rs1 == forward_reg:
                            data_rs1 = forward_data

                        if rs2 == forward_reg:
                            data_rs2 = forward_data

                        # BEQ
                        if funct3 == 0b000:
                            if data_rs1 == data_rs2:
                                self.state.IF["PC"] = self.state.EX["PC"]
                                self.state.IF["bubble"] = True

                        # BNE
                        else:
                            if data_rs1 != data_rs2:
                                self.state.IF["PC"] = self.state.EX["PC"]
                                self.state.IF["bubble"] = True

                    # LW
                    elif self.state.EX["alu_op"] == 0b0000011:
                        imm = self.state.EX["imm"]
                        rd = self.state.EX["rd"]
                        rs1 = self.state.EX["rs1"]
                        rs1_value = self.myRF.readRF(rs1)
                        forward_reg = self.state.EX["forward_reg"]
                        forward_data = self.state.EX["forward_data"]

                        if rs1 == forward_reg:
                            rs1_value = forward_data

                        rd_mem = rs1_value + twos_comp(val=imm, sign_bit=11)

                        # store memory address rd_mem
                        self.nextState.MEM["rd_mem"] = rd_mem
                        # record rd register address in the next state for MEM stage
                        self.nextState.MEM["wrt_reg_addr"] = rd
                        # # update forward_reg for checking if the next instruction uses the rd register
                        self.nextState.EX["forward_reg"] = rd

                        self.nextState.MEM["read_enable"] = True

                    # SW
                    elif self.state.EX["alu_op"] == 0b0100011:
                        # get imm
                        imm = self.state.EX["imm"]
                        rs1 = self.state.EX["rs1"]
                        rs2 = self.state.EX["rs2"]
                        rs1_value = self.myRF.readRF(rs1)
                        rs2_value = self.myRF.readRF(rs2)
                        forward_reg = self.state.EX["forward_reg"]
                        forward_data = self.state.EX["forward_data"]

                        if rs1 == forward_reg:
                            rs1_value = forward_data

                        if rs2 == forward_reg:
                            rs2_value = forward_data

                        wrt_mem = rs1_value + twos_comp(val=imm, sign_bit=11)
                        wrt_data = rs2_value

                        self.nextState.MEM["Store_data"] = wrt_data
                        self.nextState.MEM["wrt_mem"] = wrt_mem

                        self.nextState.MEM["wrt_enable"] = True
                        self.nextState.EX["forward_reg"] = rs2

            # --------------------- ID stage ---------------------
            if stage == 'ID':
                self.nextState.STAGES.append('EX')
                opcode = self.state.ID["Instr"] & (2 ** 7 - 1)

                if self.state.IF["nop"]:
                    self.nextState.ID["nop"] = True
                    self.nextState.IF["nop"] = True
                elif not self.state.IF["bubble"]:
                    # R-type
                    if opcode == 0b0110011:
                        # get funct7
                        funct7 = self.state.ID["Instr"] >> 25
                        # get funct3
                        funct3 = (self.state.ID["Instr"] >> 12) & ((1 << 3) - 1)
                        # get rs2
                        rs2 = (self.state.ID["Instr"] >> 20) & ((1 << 5) - 1)
                        # get rs1
                        rs1 = (self.state.ID["Instr"] >> 15) & ((1 << 5) - 1)
                        # get rd
                        rd = (self.state.ID["Instr"] >> 7) & ((1 << 5) - 1)

                        if self.nextState.MEM["read_enable"] and (
                                self.nextState.MEM["wrt_reg_addr"] == rs1 or self.nextState.MEM["wrt_reg_addr"] == rs2):
                            self.state.IF["bubble"] = True

                        self.nextState.EX["rs1"] = rs1
                        self.nextState.EX["rs2"] = rs2
                        self.nextState.EX["rd"] = rd
                        self.nextState.EX["alu_op"] = opcode
                        self.nextState.EX["funct3"] = funct3
                        self.nextState.EX["funct7"] = funct7

                    # I Type
                    elif opcode == 0b0010011:

                        # get immediate
                        imm = self.state.ID["Instr"] >> 20 & ((1 << 12) - 1)

                        # get funct3
                        funct3 = (self.state.ID["Instr"] >> 12) & ((1 << 3) - 1)
                        # get rs1
                        rs1 = (self.state.ID["Instr"] >> 15) & ((1 << 5) - 1)
                        # get rd
                        rd = (self.state.ID["Instr"] >> 7) & ((1 << 5) - 1)

                        if self.nextState.MEM["read_enable"] and self.nextState.MEM["wrt_reg_addr"] == rs1:
                            self.state.IF["bubble"] = True

                        self.nextState.EX["rs1"] = rs1
                        self.nextState.EX["rd"] = rd
                        self.nextState.EX["funct3"] = funct3
                        self.nextState.EX["imm"] = imm
                        self.nextState.EX["alu_op"] = opcode

                    # J Type
                    elif opcode == 0b1101111:

                        self.state.IF["bubble"] = True

                        # get imm
                        imm19_12 = (self.state.ID["Instr"] >> 12) & ((1 << 8) - 1)
                        imm11 = (self.state.ID["Instr"] >> 20) & 1
                        imm10_1 = (self.state.ID["Instr"] >> 21) & ((1 << 10) - 1)
                        imm20 = (self.state.ID["Instr"] >> 31) & 1
                        imm = (imm20 << 20) | (imm10_1 << 1) | (imm11 << 11) | (imm19_12 << 12)

                        # get rd
                        rd = (self.state.ID["Instr"] >> 7) & ((1 << 5) - 1)

                        self.nextState.EX["imm"] = imm
                        # self.nextState.ID["PC"] = self.state.IF["PC"]
                        self.state.IF["PC"] = self.state.ID["PC"] + twos_comp(val=imm, sign_bit=20)
                        self.nextState.EX["rd"] = rd
                        self.nextState.EX["alu_op"] = opcode

                    # B Type
                    elif opcode == 0b1100011:

                        self.state.IF["bubble"] = True

                        # get imm
                        imm11 = (self.state.ID["Instr"] >> 7) & 1
                        imm4_1 = (self.state.ID["Instr"] >> 8) & ((1 << 4) - 1)
                        imm10_5 = (self.state.ID["Instr"] >> 25) & ((1 << 6) - 1)
                        imm12 = (self.state.ID["Instr"] >> 31) & 1
                        imm = (imm11 << 11) | (imm4_1 << 1) | (imm10_5 << 5) | (imm12 << 12)

                        # get rs2
                        rs2 = (self.state.ID["Instr"] >> 20) & ((1 << 5) - 1)
                        # get rs1
                        rs1 = (self.state.ID["Instr"] >> 15) & ((1 << 5) - 1)
                        # get funct3
                        funct3 = (self.state.ID["Instr"] >> 12) & ((1 << 3) - 1)

                        if self.nextState.MEM["read_enable"] and (
                                self.nextState.MEM["wrt_reg_addr"] == rs1 or self.nextState.MEM["wrt_reg_addr"] == rs2):
                            self.state.IF["bubble"] = True

                        self.nextState.EX["imm"] = imm
                        self.nextState.EX["PC"] = self.state.ID["PC"] + twos_comp(val=imm, sign_bit=12)
                        self.nextState.EX["rs2"] = rs2
                        self.nextState.EX["rs1"] = rs1
                        self.nextState.EX["alu_op"] = opcode
                        self.nextState.EX["funct3"] = funct3

                    # LW
                    elif opcode == 0b0000011:

                        # get imm
                        imm = self.state.ID["Instr"] >> 20
                        # get funct3
                        funct3 = self.state.ID["Instr"] & (((1 << 3) - 1) << 12)
                        # get rs1
                        rs1 = (self.state.ID["Instr"] >> 15) & ((1 << 5) - 1)
                        # get rd
                        rd = (self.state.ID["Instr"] >> 7) & ((1 << 5) - 1)

                        self.nextState.EX["imm"] = imm
                        self.nextState.EX["rd"] = rd
                        self.nextState.EX["rs1"] = rs1
                        self.nextState.EX["alu_op"] = opcode
                        self.nextState.EX["funct3"] = funct3

                    # SW
                    elif opcode == 0b0100011:

                        # get imm
                        imm11_5 = self.state.ID["Instr"] >> 25
                        imm4_0 = (self.state.ID["Instr"] >> 7) & ((1 << 5) - 1)
                        imm = (imm11_5 << 5) | imm4_0

                        # get funct3
                        funct3 = self.state.ID["Instr"] & (((1 << 3) - 1) << 12)
                        # get rs1
                        rs1 = (self.state.ID["Instr"] >> 15) & ((1 << 5) - 1)
                        # get rd
                        rs2 = (self.state.ID["Instr"] >> 20) & ((1 << 5) - 1)

                        if self.nextState.MEM["read_enable"] and (
                                self.nextState.MEM["wrt_reg_addr"] == rs1 or self.nextState.MEM["wrt_reg_addr"] == rs2):
                            self.state.IF["bubble"] = True

                        self.nextState.EX["imm"] = imm
                        self.nextState.EX["rs2"] = rs2
                        self.nextState.EX["rs1"] = rs1
                        self.nextState.EX["alu_op"] = opcode
                        self.nextState.EX["funct3"] = funct3

            # --------------------- IF stage ---------------------
            if stage == 'IF':
                if not self.state.IF["bubble"]:
                    instr = int(self.ext_imem.readInstr(self.state.IF["PC"]), 16)
                    if instr == 0xFFFFFFFF:
                        self.nextState.IF["nop"] = True
                    else:
                        self.nextState.ID["Instr"] = instr

                    self.nextState.STAGES.append('ID')
                else:
                    self.nextState.STAGES.append('IF')

                self.nextState.ID["PC"] = self.state.IF["PC"]

        self.nextState.IF["PC"] = self.state.IF["PC"]

        if not self.nextState.IF["nop"] and not self.state.IF["bubble"] and self.state.IF["PC"] + 4 < len(
                self.ext_imem.IMem):

            if len(self.state.STAGES) > 0:
                self.nextState.IF["PC"] = self.state.IF["PC"] + 4

            self.nextState.STAGES.append('IF')

        self.nextState.IF["bubble"] = self.state.IF["bubble"]

        if self.state.IF["nop"] and self.state.ID["nop"] and self.state.EX["nop"] and self.state.MEM["nop"] and \
                self.state.WB["nop"]:
            self.halted = True

        self.myRF.outputRF(self.cycle)  # dump RF
        if self.cycle > 0:
            self.printState(self.nextState,
                            self.cycle - 1)  # print states after executing cycle 0, cycle 1, cycle 2 ...

        # The end of the cycle and updates the current state with the values calculated in this cycle
        self.state = deepcopy(self.nextState)
        self.nextState = State()
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-" * 70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.items()])
        printstate.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.items()])
        printstate.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.items()])
        printstate.extend(["MEM." + key + ": " + str(val) + "\n" for key, val in state.MEM.items()])
        printstate.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.items()])

        if (cycle == 0):
            perm = "w"
        else:
            perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


if __name__ == "__main__":

    # parse arguments for input file location
    parser = argparse.ArgumentParser(description='RV32I processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()

    ioDir = os.path.abspath(args.iodir)
    print("IO Directory:", ioDir)

    imem = InsMem("Imem", ioDir)
    dmem_ss = DataMem("SS", ioDir)
    dmem_fs = DataMem("FS", ioDir)

    ssCore = SingleStageCore(ioDir, imem, dmem_ss)
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    printList = []
    while (True):
        if not ssCore.halted:
            ssCore.step()

        if not fsCore.halted:
            fsCore.step()

        if (ssCore.halted and fsCore.halted) or fsCore.cycle > 100:
            break

    # dump SS and FS data mem.
    dmem_ss.outputDataMem()
    dmem_fs.outputDataMem()

    print("Stats for SingleStageCore")
    print("The number of Cycles spent on SingleStageCore:", ssCore.cycle, end=", ")
    print("The number of Instruction executed on SingleStageCore:", int(len(imem.IMem) / 4), end="\n\n")
    print("Stats for FiveStageCore")
    print("The number of Cycles spent on FiveStageCore:", fsCore.cycle, end=", ")
    print("The number of Instruction executed on SingleStageCore:", int(len(imem.IMem) / 4))
