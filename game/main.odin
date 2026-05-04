package game

import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"

width :: 7
height :: 6

p_width :: width + 2
p_height :: height + 2

// Changed to f32 so it is immediately ready to feed into your Neural Network
gameboard :: [2][p_height][p_width]u8

// ANSI Color Constants
ANSI_RESET :: "\x1b[0m"
ANSI_RED :: "\x1b[31m"
ANSI_YELLOW :: "\x1b[33m"
ANSI_GREEN :: "\x1b[32m"

Win_State :: struct {
	found:  bool,
	player: int,
	coords: [4][2]int,
}

Training_Sample :: struct {
	board: gameboard,
	state: Win_State,
}

check_win :: proc(board: ^gameboard) -> Win_State {
	for p in 0 ..< 2 {
		// Horizontal
		for r in 1 ..= 6 {
			for c in 1 ..= 4 {
				if board[p][r][c] == 1.0 &&
				   board[p][r][c + 1] == 1.0 &&
				   board[p][r][c + 2] == 1.0 &&
				   board[p][r][c + 3] == 1.0 {
					return Win_State {
						found = true,
						player = p,
						coords = [4][2]int{{r, c}, {r, c + 1}, {r, c + 2}, {r, c + 3}},
					}
				}
			}
		}

		// Vertical
		for r in 1 ..= 3 {
			for c in 1 ..= 7 {
				if board[p][r][c] == 1.0 &&
				   board[p][r + 1][c] == 1.0 &&
				   board[p][r + 2][c] == 1.0 &&
				   board[p][r + 3][c] == 1.0 {
					return Win_State {
						found = true,
						player = p,
						coords = [4][2]int{{r, c}, {r + 1, c}, {r + 2, c}, {r + 3, c}},
					}
				}
			}
		}

		// Diagonal Right (\)
		for r in 1 ..= 3 {
			for c in 1 ..= 4 {
				if board[p][r][c] == 1.0 &&
				   board[p][r + 1][c + 1] == 1.0 &&
				   board[p][r + 2][c + 2] == 1.0 &&
				   board[p][r + 3][c + 3] == 1.0 {
					return Win_State {
						found = true,
						player = p,
						coords = [4][2]int{{r, c}, {r + 1, c + 1}, {r + 2, c + 2}, {r + 3, c + 3}},
					}
				}
			}
		}

		// Diagonal Left (/)
		for r in 1 ..= 3 {
			for c in 4 ..= 7 {
				if board[p][r][c] == 1.0 &&
				   board[p][r + 1][c - 1] == 1.0 &&
				   board[p][r + 2][c - 2] == 1.0 &&
				   board[p][r + 3][c - 3] == 1.0 {
					return Win_State {
						found = true,
						player = p,
						coords = [4][2]int{{r, c}, {r + 1, c - 1}, {r + 2, c - 2}, {r + 3, c - 3}},
					}
				}
			}
		}
	}
	return Win_State{found = false}
}

turn :: proc(board: ^gameboard, col: int, player: int) -> bool {
	if col < 0 || col > 6 do return false
	if player < 0 || player > 1 do return false

	padded_col := col + 1

	if board[0][6][padded_col] != 0 || board[1][6][padded_col] != 0 {
		return false
	}

	for r := 1; r <= 6; r += 1 {
		if board[0][r][padded_col] == 0 && board[1][r][padded_col] == 0 {
			board[player][r][padded_col] = 1.0
			return true
		}
	}
	return false
}

random_turn :: proc(player: int, board: ^gameboard) {
	for {
		pos := rand.int_max(7) // int_max(7) picks 0 through 6 cleanly
		if turn(board, pos, player) {
			break // Valid move found, exit loop
		}
	}
}

print_board :: proc(board: ^gameboard, win_state: Win_State = {}) {
	fmt.println("---------------")
	for r := 6; r >= 1; r -= 1 {
		fmt.print("| ")
		for c := 1; c <= 7; c += 1 {
			is_winning_piece := false
			if win_state.found {
				for i in 0 ..< 4 {
					if win_state.coords[i][0] == r && win_state.coords[i][1] == c {
						is_winning_piece = true
						break
					}
				}
			}

			symbol := "."
			color := ANSI_RESET

			if board[0][r][c] == 1.0 {
				symbol = "O"
				color = is_winning_piece ? ANSI_GREEN : ANSI_RED
			} else if board[1][r][c] == 1.0 {
				symbol = "X"
				color = is_winning_piece ? ANSI_GREEN : ANSI_YELLOW
			}

			fmt.printf("%s%s%s ", color, symbol, ANSI_RESET)
		}
		fmt.println("|")
	}
	fmt.println("---------------")
}

play_game :: proc() -> (gameboard, Win_State) {
	_gameboard: gameboard = {}
	final_state: Win_State = {}

	for turn in 0 ..< 42 {
		current_player := turn % 2
		random_turn(current_player, &_gameboard)
		win_state := check_win(&_gameboard)

		if win_state.found {
			final_state = win_state
			break
		}
	}
	return _gameboard, final_state
}

// Dumps an array of game states to a raw binary file.
dump_dataset :: proc(filepath: string, samples: []Training_Sample) -> bool {
	perms: os.Permissions = {.Read_User, .Write_User, .Read_Group, .Read_Other}
	fd, err := os.open(filepath, os.O_APPEND | os.O_CREATE | os.O_WRONLY, perms)

	if err != os.ERROR_NONE {
		fmt.printfln("Failed to open dataset file: %v", err)
		return false
	}
	defer os.close(fd)

	raw_bytes := mem.slice_data_cast([]u8, samples)

	bytes_written, write_err := os.write(fd, raw_bytes)
	if write_err != os.ERROR_NONE {
		fmt.printfln("Failed to write to file: %v", write_err)
		return false
	}
	return true
}

main :: proc() {
	game, result := play_game()

	if result.found {
		fmt.printfln("Game Result: Player %v wins!", result.player)
	} else {
		fmt.println("Game Result: Draw")
	}

	print_board(&game, result)
}
