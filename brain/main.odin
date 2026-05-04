package brain

import "../game/"
import "core:fmt"
import "core:math"
import "core:math/rand"
width :: 7
height :: 6

p_width :: width + 2
p_height :: height + 2

filter_count :: 16
filter_layers :: 2
filter_wh :: 3

Vec2i :: [2]int
Snapshot :: [3][3]u8

CONV_WEIGHTS :: [filter_count][filter_layers][filter_wh][filter_wh]f32
DENSE_WEIGHTS :: [1][filter_count * width * height]f32
global: struct {
	conv:  CONV_WEIGHTS,
	dense: DENSE_WEIGHTS,
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

seed_weights :: proc() -> (CONV_WEIGHTS, DENSE_WEIGHTS) {
	mean: f32 = 0.0
	conv_fan_in: f32 = (filter_layers * filter_wh * filter_wh)
	sd := math.sqrt_f32(2.0 / conv_fan_in)
	conv_weights: CONV_WEIGHTS = {}
	for f in 0 ..< filter_count {
		for l in 0 ..< filter_layers {
			for i in 0 ..< filter_wh {
				for j in 0 ..< filter_wh {
					conv_weights[f][l][i][j] = rand.float32_normal(mean, sd)
				}
			}
		}
	}

	dense_weights: DENSE_WEIGHTS = {}
	dense_fan_in: f32 = len(dense_weights[0])
	sd = math.sqrt_f32(2.0 / dense_fan_in)
	for i in 0 ..< len(dense_weights[0]) {
		dense_weights[0][i] = rand.float32_normal(mean, sd)
	}

	return conv_weights, dense_weights
}

snapshot :: proc(board: game.gameboard, player: int, pos: Vec2i) -> Snapshot {
	snap: Snapshot = {}
	if is_valid_move(pos) {
		for r in 0 ..< 3 {
			for c in 0 ..< 3 {
				board_x := pos.x + (r - 1)
				board_y := pos.y + (c - 1)
				snap[r][c] = board[player][board_x][board_y]
			}
		}
	}
	return snap
}

is_valid_move :: proc(pos: Vec2i) -> bool {
	// The valid play area is exactly inside the 1-cell padding
	valid_x := pos.x >= 1 && pos.x <= game.width
	valid_y := pos.y >= 1 && pos.y <= game.height

	return valid_x && valid_y
}

train_step :: proc(conv: ^CONV_WEIGHTS, dense: ^DENSE_WEIGHTS, board_data: game.gameboard) {
	for p in 0 ..< filter_layers {
		for r in 1 ..< game.p_width - 1 {
			for c in 1 ..< game.p_height - 1 {
				fmt.printfln("%v", board_data[p][r][c])
			}
		}
	}
}

main :: proc() {
	seed_weights()
}
