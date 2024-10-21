package build

import (
	"blockEmulator/consensus_shard/pbft_all"
	"blockEmulator/params"
	"blockEmulator/supervisor"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"golang.org/x/exp/rand"
)

func calculateDistance(coord1, coord2 [2]uint64) float64 {
	xDiff := float64(coord1[0] - coord2[0])
	yDiff := float64(coord1[1] - coord2[1])
	return math.Sqrt(xDiff*xDiff + yDiff*yDiff)
}

type Node struct {
	Coordinates [2]uint64 `json:"coordinates"`
	Reputation  string    `json:"reputation"`
	IP          string    `json:"ip"`
}

func saveDistancesToFile(filePath string, distances map[string]map[string]float64) error {
	// 将 distances 转换为 JSON 格式
	jsonData, err := json.MarshalIndent(distances, "", "  ")
	if err != nil {
		return err
	}

	// 打开文件
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	// 将 JSON 数据写入文件
	_, err = file.Write(jsonData)
	if err != nil {
		return err
	}

	fmt.Println("Distances have been written to file:", filePath)
	return nil
}

func initConfig(nid, sid, snm uint64) *params.ChainConfig {
	params.ShardNum = int(snm)

	params.ShardNum = int(snm)
	rand.Seed(uint64(time.Now().UnixNano()))
	coordinates := make(map[string][2]uint64) // 坐标map
	params.Point_bel = make(map[uint64]map[string]uint64)
	// 读取节点信息文件
	nodeInfoData, err := os.ReadFile("NodeInfo.json")
	if err != nil {
		panic(err)
	}
	nodeInfo := make(map[string]struct {
		Coordinates [2]uint64
		IP          string
		believe     string
	})
	if err := json.Unmarshal(nodeInfoData, &nodeInfo); err != nil {
		panic(err)
	}
	// 遍历分片，并初始化节点IP和坐标
	for i := uint64(0); i < snm; i++ {
		nodesInCurrentShard := params.Nnm[i] // 分片i的节点个数
		if _, ok := params.IPmap_nodeTable[i]; !ok {
			params.IPmap_nodeTable[i] = make(map[uint64]string)
		}
		if _, exists := params.Point_bel[i]; !exists {
			params.Point_bel[i] = make(map[string]uint64)
		}
		for j := uint64(0); j < nodesInCurrentShard; j++ {
			key := fmt.Sprintf("%02d%02d", i, j)
			if node, ok := nodeInfo[key]; ok {
				params.IPmap_nodeTable[i][j] = node.IP
				coordinates[params.IPmap_nodeTable[i][j]] = node.Coordinates

			}
		}
	}

	// Initialize the distances map using IP-to-ID mapping
	params.Distances = make(map[string]map[string]float64)
	maxDistance := float64(100) // 设置距离上限为 100
	for i := uint64(0); i < 5; i++ {
		for j := uint64(0); j < params.Nnm[i]; j++ {
			if _, exists := params.Distances[params.IPmap_nodeTable[i][j]]; !exists {
				// 如果 `params.Distances[node]` 尚未初始化，则初始化它
				params.Distances[params.IPmap_nodeTable[i][j]] = make(map[string]float64)
			}
			for p := uint64(0); p < 5; p++ {
				for q := uint64(0); q < params.Nnm[p]; q++ {
					if params.IPmap_nodeTable[i][j] == params.IPmap_nodeTable[p][q] {
						params.Distances[params.IPmap_nodeTable[i][j]][params.IPmap_nodeTable[i][j]] = 0
					} else {
						distance := calculateDistance(coordinates[params.IPmap_nodeTable[i][j]], coordinates[params.IPmap_nodeTable[p][q]])
						if distance > maxDistance {
							distance = maxDistance // 如果距离大于 100，设置为 100
						}
						params.Distances[params.IPmap_nodeTable[i][j]][params.IPmap_nodeTable[p][q]] = distance
					}
				}
			}

		}

	}

	// 保存 distances 到文件
	err1 := saveDistancesToFile("distances.json", params.Distances)
	if err1 != nil {
		fmt.Println("Error saving distances:", err1)
	}
	
	// Supervisor node setup
	if _, ok := params.IPmap_nodeTable[params.DeciderShard]; !ok {
		params.IPmap_nodeTable[params.DeciderShard] = make(map[uint64]string)
	}
	params.IPmap_nodeTable[params.DeciderShard][0] = params.SupervisorAddr

	// Create ChainConfig instance
	pcc := &params.ChainConfig{
		ChainID:       sid,
		NodeID:        nid,
		ShardID:       sid,
		ShardNums:     snm,
		BlockSize:     uint64(params.MaxBlockSize_global),
		BlockInterval: uint64(params.Block_Interval),
		InjectSpeed:   uint64(params.InjectSpeed),
		Coordinates:   coordinates, // 现在 `coordinates` 的类型是 `map[uint64][2]uint64]`，匹配 ChainConfig
	}
	for outerKey, innerMap := range params.Point_bel {
		fmt.Printf("Outer Key: %d\n", outerKey)
		nodepersnm := 0
		for innerKey, value := range innerMap {
			fmt.Printf("  Inner Key: %s, Value: %d\n", innerKey, value)
			if innerMap[innerKey] == 1 {
				nodepersnm++
			}

		}
		fmt.Print(float64(nodepersnm) / float64(params.Nnm[outerKey]))
		fmt.Print("\n")
	}

	return pcc
}
func BuildSupervisor(snm, mod uint64) {
	var measureMod []string
	if mod == 0 || mod == 2 {
		measureMod = params.MeasureBrokerMod
	} else {
		measureMod = params.MeasureRelayMod
	}
	measureMod = append(measureMod, "Tx_Details")

	lsn := new(supervisor.Supervisor)
	lsn.NewSupervisor(params.SupervisorAddr, initConfig(123, 123, snm), params.CommitteeMethod[mod], measureMod...)
	time.Sleep(10000 * time.Millisecond)
	go lsn.SupervisorTxHandling()
	lsn.TcpListen()
}

func BuildNewPbftNode(nid, sid, snm, mod uint64) {
	worker := pbft_all.NewPbftNode(sid, nid, initConfig(nid, sid, snm), params.CommitteeMethod[mod])
	go worker.TcpListen()
	worker.Propose()
}
