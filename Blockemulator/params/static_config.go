package params

import "math/big"

type ChainConfig struct {
	ChainID       uint64
	NodeID        uint64
	ShardID       uint64
	ShardNums     uint64
	BlockSize     uint64
	BlockInterval uint64
	InjectSpeed   uint64

	// used in transaction relaying, useless in brokerchain mechanism
	MaxRelayBlockSize uint64
	Coordinates       map[string][2]uint64 // 修改为uint64类型

}

// 定义全局变量 nnm，表示每个分片的节点数量
var Nnm = []uint64{8, 34, 16, 4, 38} // 示例：分片0有3个节点，分片1有5个节点，分片2有4个节点
var (
	DeciderShard     = uint64(0xffffffff)
	Init_Balance, _  = new(big.Int).SetString("100000000000000000000000000000000000000000000", 10)
	IPmap_nodeTable  = make(map[uint64]map[uint64]string)
	CommitteeMethod  = []string{"CLPA_Broker", "CLPA", "Broker", "Relay"}
	MeasureBrokerMod = []string{"TPS_Broker", "TCL_Broker", "CrossTxRate_Broker", "TxNumberCount_Broker"}
	MeasureRelayMod  = []string{"TPS_Relay", "TCL_Relay", "CrossTxRate_Relay", "TxNumberCount_Relay"}
)

var Distances map[string]map[string]float64
var Point_bel map[uint64]map[string]uint64
