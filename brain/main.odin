package brain

import "../game/"
import "../helpers"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:os"

// --- Constants ---

filter_count :: helpers.filter_count
filter_layers :: helpers.filter_layers
filter_wh :: helpers.filter_wh

// --- Types ---

Vec2i :: helpers.Vec2i
Kernel :: helpers.Kernel

L1_CONV_WEIGHTS :: helpers.L1_CONV_WEIGHTS
LX_CONV_WEIGHTS :: helpers.LX_CONV_WEIGHTS
DENSE_WEIGHTS :: helpers.DENSE_WEIGHTS
FLATTENED_MAP :: helpers.FLATTENED_MAP
FEATURE_MAP :: helpers.FEATURE_MAP

NETWORK_WEIGHTS :: helpers.NETWORK_WEIGHTS
CHAMPION :: NETWORK_WEIGHTS
CHALLENGER :: NETWORK_WEIGHTS

// --- Dataset ---

// Reads the binary file and reconstructs the array of Training_Samples.
load_dataset :: proc(
	filepath: string,
	aloc: runtime.Allocator,
) -> (
	samples: []helpers.Training_Sample,
	ok: bool,
) {
	// 1. Read the entire file into a raw byte array ([]u8)
	file_data, success := os.read_entire_file(filepath, aloc)
	if success != nil {
		fmt.printfln("Failed to read dataset file: %v", filepath)
		return nil, false
	}

	samples = mem.slice_data_cast([]helpers.Training_Sample, file_data)

	return samples, true
}

// --- Weight Initialization ---

seed_weights :: proc(alloc: runtime.Allocator) -> ^NETWORK_WEIGHTS {
	weights := new(NETWORK_WEIGHTS, alloc)
	mean: f32 = 0.0

	// --- 1. INITIALIZE LAYER 1 ---
	// fan_in = 2 channels * 3 width * 3 height
	l1_fan_in: f32 = 2.0 * f32(filter_wh * filter_wh)
	l1_sd := math.sqrt_f32(2.0 / l1_fan_in)

	for f in 0 ..< filter_count {
		for l in 0 ..< 2 {
			for i in 0 ..< filter_wh {
				for j in 0 ..< filter_wh {
					weights.conv1[f][l][i][j] = rand.float32_normal(mean, l1_sd)
				}
			}
		}
	}

	// --- 2. INITIALIZE LAYERS 2 & 3 ---
	// fan_in = 16 channels * 3 width * 3 height
	lx_fan_in: f32 = f32(filter_count * filter_wh * filter_wh)
	lx_sd := math.sqrt_f32(2.0 / lx_fan_in)

	for f in 0 ..< filter_count {
		for l in 0 ..< filter_count {
			for i in 0 ..< filter_wh {
				for j in 0 ..< filter_wh {
					// We can do both at the same time since they share dimensions
					weights.conv2[f][l][i][j] = rand.float32_normal(mean, lx_sd)
					weights.conv3[f][l][i][j] = rand.float32_normal(mean, lx_sd)
				}
			}
		}
	}

	// --- 3. INITIALIZE DENSE LAYER ---
	dense_fan_in: f32 = f32(len(weights.dense[0]))
	dense_sd := math.sqrt_f32(2.0 / dense_fan_in)
	for i in 0 ..< len(weights.dense[0]) {
		weights.dense[0][i] = rand.float32_normal(mean, dense_sd)
	}

	for col in 0 ..< 7 {
		for i in 0 ..< len(weights.dense_pol[0]) {
			weights.dense_pol[col][i] = rand.float32_range(mean, dense_sd)
		}
	}

	return weights
}

// --- Forward Pass ---

convolution_l1 :: proc(board: helpers.gameboard, weights: ^L1_CONV_WEIGHTS) -> FEATURE_MAP {
	output: FEATURE_MAP

	for k in 0 ..< filter_count {
		for r in 0 ..< helpers.height {
			for c in 0 ..< helpers.width {
				sum: f32 = 0.0

				// Layer 1 only loops through 2 incoming layers
				for layer in 0 ..< 2 {
					for kr in 0 ..< filter_wh {
						for kc in 0 ..< filter_wh {
							weight := weights[k][layer][kr][kc]
							board_val := board[layer][r + kr][c + kc]
							sum += weight * f32(board_val)
						}
					}
				}

				// Write to r+1 and c+1 to keep it centered inside the padding!
				output[k][r + 1][c + 1] = max(0.0, sum)
			}
		}
	}
	return output
}

convolution_lx :: proc(
	input: FEATURE_MAP, // Takes the output of the previous layer
	weights: ^LX_CONV_WEIGHTS,
) -> FEATURE_MAP {
	output: FEATURE_MAP

	for k in 0 ..< filter_count {
		for r in 0 ..< helpers.height {
			for c in 0 ..< helpers.width {
				sum: f32 = 0.0

				// Hidden layers must loop through all 16 incoming feature maps
				for layer in 0 ..< filter_count {
					for kr in 0 ..< filter_wh {
						for kc in 0 ..< filter_wh {
							weight := weights[k][layer][kr][kc]

							// No need to cast to f32 here, the input is already f32
							in_val := input[layer][r + kr][c + kc]
							sum += weight * in_val
						}
					}
				}

				output[k][r + 1][c + 1] = max(0.0, sum)
			}
		}
	}
	return output
}

flatten :: proc(l3_out: ^FEATURE_MAP) -> ^FLATTENED_MAP {
	flat_input := transmute(^FLATTENED_MAP)l3_out
	return flat_input
}

calculate_z :: proc(flattened_map: ^FLATTENED_MAP, dense: ^DENSE_WEIGHTS) -> f32 {
	z: f32 = 0.0
	for i in 0 ..< len(flattened_map) {
		weight_val := dense[0][i]
		flat_val := flattened_map^[i]

		z = z + (weight_val * flat_val)
	}
	return z
}

value_head :: proc(z: f32) -> f32 {
	return math.tanh(z)
}

policy_head :: proc(
	flattened_map: ^FLATTENED_MAP,
	policy_weights: ^[7][len(flattened_map^)]f32,
) -> [7]f32 {
	logits: [7]f32
	probs: [7]f32

	// Raw 7 logits
	for col in 0 ..< 7 {
		z: f32 = 0.0
		for i in 0 ..< len(flattened_map^) {
			z += policy_weights[col][i] * flattened_map^[i]
		}
		logits[col] = z
	}

	// Find highest logits
	max_logit: f32 = 0.0
	for i in 1 ..< 7 {
		if logits[i] > max_logit {
			max_logit = logits[i]
		}
	}

	// Exponentiate and sum
	sum_exp: f32 = 0.0
	for i in 0 ..< 7 {
		// Subract max logit so we don't overflow
		exp_val := math.exp_f32(logits[i] - max_logit)
		probs[i] = exp_val
		sum_exp += exp_val
	}

	for i in 0 ..< 7 {
		probs[i] = probs[i] / sum_exp
	}

	return probs
}

// --- Loss ---

/*
 * z is the actual final result (1 if it won, -1 if it lost, 0 if it drew).
 * v is what the network predicted on that specific turn (e.g., 0.6).
 */
value_loss :: proc(v: f32, z_true: int) -> f32 {
	lv := v - f32(z_true)
	lv_sqr := lv * lv
	return lv_sqr
}

policy_loss :: proc(policy_val: [7]f32, z_true: int) -> f32 {
	lp: f32 = 0.0
	for i in 0 ..< 7 {
		lp = lp + (-1 * math.log_f32(policy_val[i], math.PI))
	}
	return lp
}

total_loss :: proc(value: f32, policy: f32) -> f32 {
	return value + policy
}

back_prop :: proc() {

}

// --- Training ---

train_step :: proc(network: ^NETWORK_WEIGHTS, data: ^helpers.Training_Sample) {
	// --- Forward Pass ---
	l1_out: FEATURE_MAP = convolution_l1(data.board, &network.conv1)
	l2_out: FEATURE_MAP = convolution_lx(l1_out, &network.conv2)
	l3_out: FEATURE_MAP = convolution_lx(l2_out, &network.conv3)

	flattened_map := flatten(&l3_out)
	fmt.printfln("L3 Size: %v", len(flattened_map))

	z := calculate_z(flattened_map, &network.dense)
	fmt.printfln("Z Sum: %v", z)


	if data.z_true == 0 {
		fmt.println("Actual Winner: Tie")
	} else if data.z_true == 1 {
		fmt.println("Actual Winner: Player 1")
	} else {
		fmt.println("Actual Winner: Player 2")
	}

	// --- Prediction/Loss ---
	fmt.println()
	fmt.println("--- Value Head ---")
	network.dense_val = value_head(z)
	loss_val := value_loss(network.dense_val, data.z_true)
	fmt.printfln("Prediction: %v", network.dense_val)
	fmt.printfln("Value Loss: %v", loss_val)

	fmt.println()
	fmt.println("--- Policy Head ---")
	policy_val := policy_head(flattened_map, &network.dense_pol)
	loss_pol := policy_loss(policy_val, data.z_true)
	fmt.printfln("Policy Distribution: %v", policy_val)
	fmt.printfln("Policy Loss: %v", loss_pol)

	fmt.println()
	fmt.println("--- Total Loss ---")
	loss_tot := total_loss(loss_val, loss_pol)
	fmt.printfln("Total Loss: %v", loss_tot)
}

// --- Board Utilities ---

is_valid_move :: proc(pos: Vec2i) -> bool {
	// The valid play area is exactly inside the 1-cell padding
	valid_x := pos.x >= 1 && pos.x <= helpers.width
	valid_y := pos.y >= 1 && pos.y <= helpers.height

	return valid_x && valid_y
}

snapshot :: proc(board: helpers.gameboard, player: int, pos: Vec2i) -> Kernel {
	k: Kernel = {}
	if is_valid_move(pos) {
		for r in 0 ..< 3 {
			for c in 0 ..< 3 {
				board_x := pos.x + (r - 1)
				board_y := pos.y + (c - 1)
				k[r][c] = board[player][board_x][board_y]
			}
		}
	}
	return k
}

// --- Entry Point ---
main :: proc() {
	alloc := context.temp_allocator
	defer free_all(alloc)

	network := seed_weights(alloc)

	dataset_file :: "connect4_training_data.bin"
	data, success := load_dataset(dataset_file, alloc)

	if !success {
		fmt.println("Error loading dataset into memory")
		return
	}
	train_step(network, &data[rand.int_range(0, 1000)])
}
