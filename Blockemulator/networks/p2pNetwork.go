package networks

import (
	"blockEmulator/params"
	"bytes"
	"io"
	"log"
	"net"
	"sort"
	"sync"
	"time"
)

var receiverShardID uint64
var receiverNodeID uint64
var senderShardID uint64
var senderNodeID uint64
var connMaplock sync.Mutex
var connectionPool = make(map[string]net.Conn, 0)

// TcpDial attempts to send the message after a delay
func TcpDial(context []byte, addr string) {
	connMaplock.Lock()
	defer connMaplock.Unlock()

	var err error
	var conn net.Conn // Define conn here
	if c, ok := connectionPool[addr]; ok {
		if tcpConn, tcpOk := c.(*net.TCPConn); tcpOk {
			if err := tcpConn.SetKeepAlive(true); err != nil {
				delete(connectionPool, addr) // Remove if not alive
				conn, err = net.Dial("tcp", addr)
				if err != nil {
					log.Println("Reconnect error", err)
					return
				}
				connectionPool[addr] = conn
				go ReadFromConn(addr) // Start reading from new connection
			} else {
				conn = c // Use the existing connection
			}
		}
	} else {
		conn, err = net.Dial("tcp", addr)
		if err != nil {
			log.Println("Connect error", err)
			return
		}
		connectionPool[addr] = conn
		go ReadFromConn(addr) // Start reading from new connection
	}

	_, err = conn.Write(append(context, '\n'))
	if err != nil {
		log.Println("Write error", err)
		return
	}
}

func Broadcast(sender string, receivers []string, msg []byte) {

	found := false

	// Find the sender's shard ID and node ID within that shard
	for shardID, addrMap := range params.IPmap_nodeTable {
		for nodeID, nodeAddr := range addrMap {
			if nodeAddr == sender {
				senderShardID = shardID
				senderNodeID = nodeID
				found = true
				break
			}
		}
		if found {
			break
		}
	}

	if !found {
		log.Fatalf("Sender IP not found in IPmap_nodeTable: %s", sender)
		return
	}

	// Check if sender IP is in distances map
	if _, ok := params.Distances[sender]; !ok {
		log.Fatalf("Sender IP not found in distances map: %s", sender)
		return
	}

	// Prepare list of receivers and their distances
	type receiverDistance struct {
		addr     string
		distance float64
	}
	var receiverDistances []receiverDistance

	for _, addr := range receivers {
		if addr == sender || addr == params.SupervisorAddr {
			// Skip if receiver is the sender or supervisor
			continue
		}

		// Find the receiver's shard ID and node ID within that shard

		foundReceiver := false
		for shardID, addrMap := range params.IPmap_nodeTable {
			for nodeID, nodeAddr := range addrMap {
				if nodeAddr == addr {
					receiverShardID = shardID
					receiverNodeID = nodeID
					foundReceiver = true
					break
				}
			}
			if foundReceiver {
				break
			}
		}

		if !foundReceiver {
			log.Printf("Receiver IP not found in IPmap_nodeTable: %s", addr)
			continue
		}

		// Check if receiver IP is in distances map
		if _, ok := params.Distances[addr]; !ok {
			log.Printf("Receiver IP not found in distances map: %s", addr)
			continue
		}

		// Use global distances array to get distance and add to list
		distance := params.Distances[sender][addr]
		receiverDistances = append(receiverDistances, receiverDistance{addr, distance})
	}

	// Sort receivers by distance
	sort.Slice(receiverDistances, func(i, j int) bool {
		return receiverDistances[i].distance < receiverDistances[j].distance
	})

	// If sender is supervisor, skip sorting
	if sender == params.SupervisorAddr {
		for _, receiver := range receivers {
			go TcpDial(msg, receiver)
		}
		return
	}

    for _, receiver := range receiverDistances {
		go func(r receiverDistance) {
            // Delay based on distance before calling TcpDial
            time.Sleep(time.Millisecond * time.Duration(r.distance * 10))
            TcpDial(msg, r.addr)
        }(receiver)
    }
}

// CloseAllConnInPool closes all connections in the connection pool.
func CloseAllConnInPool() {
	connMaplock.Lock()
	defer connMaplock.Unlock()

	for _, conn := range connectionPool {
		conn.Close()
	}
	connectionPool = make(map[string]net.Conn) // Reset the pool
}

// ReadFromConn reads data from a connection.
func ReadFromConn(addr string) {
	conn := connectionPool[addr]

	buffer := make([]byte, 1024)
	var messageBuffer bytes.Buffer

	for {
		n, err := conn.Read(buffer)
		if err != nil {
			if err != io.EOF {
				log.Println("Read error for address", addr, ":", err)
			}
			break
		}

		// add message to buffer
		messageBuffer.Write(buffer[:n])

		// handle the full message
		for {
			message, err := readMessage(&messageBuffer)
			if err == io.ErrShortBuffer {
				// continue to load if buffer is short
				break
			} else if err == nil {
				// log the full message
				log.Println("Received from", addr, ":", message)
			} else {
				// handle other errs
				log.Println("Error processing message for address", addr, ":", err)
				break
			}
		}
	}
}

func readMessage(buffer *bytes.Buffer) (string, error) {
	message, err := buffer.ReadBytes('\n')
	if err != nil && err != io.EOF {
		return "", err
	}
	return string(message), nil
}
