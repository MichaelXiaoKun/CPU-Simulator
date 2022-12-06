import unittest
# from unittest import TestCase
import os
import argparse

from CPU import InsMem, DataMem, RegisterFile


def get_ioDir(folder):
    parser = argparse.ArgumentParser("Start Testing " + folder)
    parser.add_argument('--iodir', default="", type=str)
    args = parser.parse_args()
    ioDir = os.path.abspath(args.iodir)

    return ioDir + "/" + folder


class InsMemTest(unittest.TestCase):

    def test_readInstr_TC0(self):
        ioDir = get_ioDir('TC0')
        instr_mem = InsMem('IMEM_TC0', ioDir)

        with open(ioDir + "/imem.txt") as im:
            instr_mem_bin8 = [data.replace("\n", "") for data in im.readlines()]

        for i in range(0, len(instr_mem_bin8), 4):
            hex8 = 0
            for j in range(i, i + 4):
                hex8 = hex8 | int('0b' + instr_mem_bin8[j], 2)
                if j < i + 3:
                    hex8 <<= 8

            # print("Case " + str(int(i / 4)) + " of TC0 in InsMemTest")
            self.assertEqual('0x' + '0' * (8 - len(hex(hex8)[2:])) + hex(hex8)[2:], instr_mem.readInstr(i))


def getResult(address, writeData, data_mem):
    data_mem.writeDataMem(address, writeData)
    return data_mem.readDataMem(address)


class DataMemTest(unittest.TestCase):

    def test_readDataMem(self):
        ioDir = get_ioDir('TC0')
        data_mem = DataMem('IMEM_TC0', ioDir)

        with open(ioDir + "/dmem.txt") as dm:
            data_mem_bin8 = [data.replace("\n", "") for data in dm.readlines()]

        for i in range(0, len(data_mem_bin8), 4):
            hex8 = 0
            for j in range(i, i + 4):
                hex8 = hex8 | int('0b' + data_mem_bin8[j], 2)
                if j < i + 3:
                    hex8 <<= 8

            self.assertEqual('0x' + '0' * (8 - len(hex(hex8)[2:])) + hex(hex8)[2:], data_mem.readDataMem(i))

    def test_writeDataMem(self):
        ioDir = get_ioDir('TC0')
        data_mem = DataMem('IMEM_TC0', ioDir)

        bin32 = getResult(0, 0x00000001, data_mem)
        self.assertEqual('0x00000001', '0x' + '0' * (8 - len(bin32[2:])) + bin32[2:])

#
# class RegisterFileTest(RegisterFile, TestCase):
#
#     def __init__(self, ioDir):
#         super().__init__(ioDir)


if __name__ == '__main__':
    unittest.main()
