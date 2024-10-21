package build

import (
	"blockEmulator/params"
	"fmt"
	"log"
	"os"
)

func GenerateBatFile(shardnum, modID int, dataRootDir string) {
	fileName := fmt.Sprintf("bat_shardNum=%v_mod=%v.bat", shardnum, params.CommitteeMethod[modID])
	ofile, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0777)
	if err != nil {
		log.Panic(err)
	}
	defer ofile.Close()

	for j := 0; j < shardnum; j++ {
		// 遍历每个分片，获取该分片的节点数量
		nodeCount := int(params.Nnm[j])
		for i := 1; i < nodeCount; i++ {
			str := fmt.Sprintf("start cmd /k go run main.go -n %d -N %d -s %d -S %d -m %d -d %s \n\n", i, nodeCount, j, shardnum, modID, dataRootDir)
			ofile.WriteString(str)
		}
	}

	for j := 0; j < shardnum; j++ {
		nodeCount := int(params.Nnm[j])
		str := fmt.Sprintf("start cmd /k go run main.go -n 0 -N %d -s %d -S %d -m %d -d %s \n\n", nodeCount, j, shardnum, modID, dataRootDir)
		ofile.WriteString(str)
	}

	str := fmt.Sprintf("start cmd /k go run main.go -c -S %d -m %d -d %s \n\n", shardnum, modID, dataRootDir)

	ofile.WriteString(str)
}

func GenerateShellFile(shardnum, modID int, dataRootDir string) {
	fileName := fmt.Sprintf("bat_shardNum=%v_mod=%v.sh", shardnum, params.CommitteeMethod[modID])
	ofile, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0777)
	if err != nil {
		log.Panic(err)
	}
	defer ofile.Close()
	ofile.WriteString("#!/bin/bash \n\n")

	for j := 0; j < shardnum; j++ {
		// 遍历每个分片，获取该分片的节点数量
		nodeCount := int(params.Nnm[j])
		for i := 1; i < nodeCount; i++ {
			str := fmt.Sprintf("go run main.go -n %d -N %d -s %d -S %d -m %d -d %s &\n\n", i, nodeCount, j, shardnum, modID, dataRootDir)
			ofile.WriteString(str)
		}
	}

	for j := 0; j < shardnum; j++ {
		nodeCount := int(params.Nnm[j])
		str := fmt.Sprintf("go run main.go -n 0 -N %d -s %d -S %d -m %d -d %s &\n\n", nodeCount, j, shardnum, modID, dataRootDir)
		ofile.WriteString(str)
	}

	str := fmt.Sprintf("go run main.go -c -S %d -m %d -d %s &\n\n", shardnum, modID, dataRootDir)

	ofile.WriteString(str)
}
func Exebat_Windows_GenerateBatFile(shardnum, modID int, dataRootDir string) {
	fileName := fmt.Sprintf("WinExe_bat_shardNum=%v_mod=%v.bat", shardnum, params.CommitteeMethod[modID])
	ofile, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0777)
	if err != nil {
		log.Panic(err)
	}
	defer ofile.Close()

	for j := 0; j < shardnum; j++ {
		nodeCount := int(params.Nnm[j]) // 获取每个分片的节点数量
		for i := 1; i < nodeCount; i++ {
			str := fmt.Sprintf("start cmd /k blockEmulator_Windows_Precompile.exe -n %d -N %d -s %d -S %d -m %d -d %s\n\n", i, nodeCount, j, shardnum, modID, dataRootDir)
			ofile.WriteString(str)
		}
	}

	for j := 0; j < shardnum; j++ {
		nodeCount := int(params.Nnm[j])
		str := fmt.Sprintf("start cmd /k blockEmulator_Windows_Precompile.exe -n 0 -N %d -s %d -S %d -m %d -d %s\n\n", nodeCount, j, shardnum, modID, dataRootDir)
		ofile.WriteString(str)
	}

	str := fmt.Sprintf("start cmd /k blockEmulator_Windows_Precompile.exe -c -S %d -m %d -d %s\n\n", shardnum, modID, dataRootDir)
	ofile.WriteString(str)
}

func Exebat_Linux_GenerateShellFile(shardnum, modID int, dataRootDir string) {
	fileName := fmt.Sprintf("bat_shardNum=%v_mod=%v.sh", shardnum, params.CommitteeMethod[modID])
	ofile, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0777)
	if err != nil {
		log.Panic(err)
	}
	defer ofile.Close()
	ofile.WriteString("#!/bin/bash \n\n")

	for j := 0; j < shardnum; j++ {
		nodeCount := int(params.Nnm[j]) // 获取每个分片的节点数量
		for i := 1; i < nodeCount; i++ {
			str := fmt.Sprintf("./blockEmulator_Linux_Precompile -n %d -N %d -s %d -S %d -m %d -d %s &\n\n", i, nodeCount, j, shardnum, modID, dataRootDir)
			ofile.WriteString(str)
		}
	}

	for j := 0; j < shardnum; j++ {
		nodeCount := int(params.Nnm[j])
		str := fmt.Sprintf("./blockEmulator_Linux_Precompile -n 0 -N %d -s %d -S %d -m %d -d %s &\n\n", nodeCount, j, shardnum, modID, dataRootDir)
		ofile.WriteString(str)
	}

	str := fmt.Sprintf("./blockEmulator_Linux_Precompile -c -S %d -m %d -d %s &\n\n", shardnum, modID, dataRootDir)
	ofile.WriteString(str)
}

func Exebat_MacOS_GenerateShellFile(shardnum, modID int, dataRootDir string) {
	fileName := fmt.Sprintf("bat_shardNum=%v_mod=%v.sh", shardnum, params.CommitteeMethod[modID])
	ofile, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0777)
	if err != nil {
		log.Panic(err)
	}
	defer ofile.Close()
	ofile.WriteString("#!/bin/bash \n\n")

	for j := 0; j < shardnum; j++ {
		nodeCount := int(params.Nnm[j]) // 获取每个分片的节点数量
		for i := 1; i < nodeCount; i++ {
			str := fmt.Sprintf("./blockEmulator_MacOS_Precompile -n %d -N %d -s %d -S %d -m %d -d %s &\n\n", i, nodeCount, j, shardnum, modID, dataRootDir)
			ofile.WriteString(str)
		}
	}

	for j := 0; j < shardnum; j++ {
		nodeCount := int(params.Nnm[j])
		str := fmt.Sprintf("./blockEmulator_MacOS_Precompile -n 0 -N %d -s %d -S %d -m %d -d %s &\n\n", nodeCount, j, shardnum, modID, dataRootDir)
		ofile.WriteString(str)
	}

	str := fmt.Sprintf("./blockEmulator_MacOS_Precompile -c -S %d -m %d -d %s &\n\n", shardnum, modID, dataRootDir)
	ofile.WriteString(str)
}
