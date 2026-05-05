package brain

import "../game/"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:os"
width :: 7
height :: 6

p_width :: width + 2
p_height :: height + 2

filter_count :: 16
filter_layers :: 2 // 2 Players
filter_wh :: 3

Vec2i :: [2]int
Kernel :: [3][3]u8

L1_CONV_WEIGHTS :: [filter_count][filter_layers][filter_wh][filter_wh]f32
LX_CONV_WEIGHTS :: [64][64][filter_wh][filter_wh]f32
DENSE_WEIGHTS :: [1][filter_count * width * height]f32

NETWORK_WEIGHTS :: struct {
	conv1: L1_CONV_WEIGHTS,
	conv2: LX_CONV_WEIGHTS,
	conv3: LX_CONV_WEIGHTS,
	dense: DENSE_WEIGHTS,
}

// Reads the binary file and reconstructs the array of Training_Samples.
load_dataset :: proc(
	filepath: string,
	aloc: runtime.Allocator,
) -> (
	samples: []game.Training_Sample,
	ok: bool,
) {
	// 1. Read the entire file into a raw byte array ([]u8)
	// Note: This allocates memory on the heap!
	file_data, success := os.read_entire_file(filepath, aloc)
	if success != nil {
		fmt.printfln("Failed to read dataset file: %v", filepath)
		return nil, false
	}

	// 2. The Odin Superpower (Reverse): Cast the bytes back into the struct array
	samples = mem.slice_data_cast([]game.Training_Sample, file_data)

	return samples, true
}

//dense_weights : []filter

// ---- MATH!!! ----

/*
 * The Loss Function fyi
 * z is the actual final result (1 if it won, -1 if it lost, 0 if it drew).
 * v is what the network predicted on that specific turn (e.g., 0.6).
 */
mean_square_error :: proc(z: f32, v: f32) -> f32 {
	mean_error := z * v
	mean_square_error := mean_error * mean_error
	return mean_square_error
}

/*
flatten :: proc(output_tensor: ^output_tensor) -> ^[magnitude]f32 {
	output_tensor := output_tensor
	flat_array := cast(^[magnitude]f32)&output_tensor
	return flat_array
}
*/

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
	// fan_in = 64 channels * 3 width * 3 height
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
	// TODO
	dense_fan_in: f32 = f32(len(weights.dense[0]))
	dense_sd := math.sqrt_f32(2.0 / dense_fan_in)

	for i in 0 ..< len(weights.dense[0]) {
		weights.dense[0][i] = rand.float32_normal(mean, dense_sd)
	}

	return weights
}

snapshot :: proc(board: game.gameboard, player: int, pos: Vec2i) -> Kernel {
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

is_valid_move :: proc(pos: Vec2i) -> bool {
	// The valid play area is exactly inside the 1-cell padding
	valid_x := pos.x >= 1 && pos.x <= game.width
	valid_y := pos.y >= 1 && pos.y <= game.height

	return valid_x && valid_y
}

train_step :: proc(network: ^NETWORK_WEIGHTS, board_data: game.gameboard) {
	l1_out: FEATURE_MAP = convolution_l1(board_data, &network.conv1)
	l2_out: FEATURE_MAP = convolution_lx(l1_out, &network.conv2)
	l3_out: FEATURE_MAP = convolution_lx(l2_out, &network.conv3)
	//TODO dense_step

}

// I highly recommend aliasing your feature map shape to keep the code clean
FEATURE_MAP :: [filter_count][p_height][p_width]f32

convolution_l1 :: proc(board: game.gameboard, weights: ^L1_CONV_WEIGHTS) -> FEATURE_MAP {
	output: FEATURE_MAP

	for k in 0 ..< filter_count {
		for r in 0 ..< height {
			for c in 0 ..< width {
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
		for r in 0 ..< height {
			for c in 0 ..< width {
				sum: f32 = 0.0

				// Hidden layers must loop through all 64 incoming feature maps
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

main :: proc() {
	alloc := context.temp_allocator
	defer free_all(alloc)

	network := seed_weights(alloc)

	fmt.printfln("Conv1 [0][0][0][0]: %f", network.conv1[0][0][0][0])
	fmt.printfln("Conv2 [0][0][0][0]: %f", network.conv2[0][0][0][0])
	fmt.printfln("Conv3 [0][0][0][0]: %f", network.conv3[0][0][0][0])
	fmt.printfln("Dense [0][0]:       %f", network.dense[0][0])

	dataset_file :: "connect4_training_data.bin"
	data, success := load_dataset(dataset_file, alloc)

	if !success {
		fmt.println("Error loading dataset into memory")
		return
	}
	train_step(network, data[0].board)
}
