package helpers

width :: 7
height :: 6

p_width :: width + 2
p_height :: height + 2

filter_count :: 16
filter_layers :: 2 // 2 Players
filter_wh :: 3

// ANSI Color Constants
ANSI_RESET :: "\x1b[0m"
ANSI_RED :: "\x1b[31m"
ANSI_YELLOW :: "\x1b[33m"
ANSI_GREEN :: "\x1b[32m"

// Types
gameboard :: [2][p_height][p_width]u8

Vec2i :: [2]int
Kernel :: [3][3]u8

L1_CONV_WEIGHTS :: [filter_count][filter_layers][filter_wh][filter_wh]f32
LX_CONV_WEIGHTS :: [filter_count][filter_count][filter_wh][filter_wh]f32
DENSE_WEIGHTS :: [1][filter_count * p_width * p_height]f32
FLATTENED_MAP :: [filter_count * p_height * p_width]f32
FEATURE_MAP :: [filter_count][p_height][p_width]f32

NETWORK_WEIGHTS :: struct {
	conv1:     L1_CONV_WEIGHTS,
	conv2:     LX_CONV_WEIGHTS,
	conv3:     LX_CONV_WEIGHTS,
	dense:     DENSE_WEIGHTS,
	dense_val: f32,
	dense_pol: [7][len(FLATTENED_MAP)]f32,
}

Win_State :: struct {
	found:  bool,
	player: int,
	coords: [4][2]int,
}

Training_Sample :: struct {
	board:     gameboard,
	pi_target: [7]f32,
	z_true:    int,
	state:     Win_State,
}

Arena :: struct {
	samples:    []Training_Sample,
	champion:   ^NETWORK_WEIGHTS,
	challenger: ^NETWORK_WEIGHTS,
}
