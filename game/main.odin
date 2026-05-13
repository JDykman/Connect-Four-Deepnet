package game

import "../helpers"
import "core:fmt"

check_win :: proc(board: helpers.gameboard) -> helpers.Win_State {
	for p in 0 ..< 2 {
		val := u8(1)
		// Horizontal
		for r in 1 ..= 6 {
			for c in 1 ..= 4 {
				if board[p][r][c] == val &&
				   board[p][r][c + 1] == val &&
				   board[p][r][c + 2] == val &&
				   board[p][r][c + 3] == val {
					return {
						found = true,
						player = p + 1,
						coords = {{r, c}, {r, c + 1}, {r, c + 2}, {r, c + 3}},
					}
				}
			}
		}

		// Vertical
		for r in 1 ..= 3 {
			for c in 1 ..= 7 {
				if board[p][r][c] == val &&
				   board[p][r + 1][c] == val &&
				   board[p][r + 2][c] == val &&
				   board[p][r + 3][c] == val {
					return {
						found = true,
						player = p + 1,
						coords = {{r, c}, {r + 1, c}, {r + 2, c}, {r + 3, c}},
					}
				}
			}
		}

		// Diagonal Right (\)
		for r in 1 ..= 3 {
			for c in 1 ..= 4 {
				if board[p][r][c] == val &&
				   board[p][r + 1][c + 1] == val &&
				   board[p][r + 2][c + 2] == val &&
				   board[p][r + 3][c + 3] == val {
					return {
						found = true,
						player = p + 1,
						coords = {{r, c}, {r + 1, c + 1}, {r + 2, c + 2}, {r + 3, c + 3}},
					}
				}
			}
		}

		// Diagonal Left (/)
		for r in 1 ..= 3 {
			for c in 4 ..= 7 {
				if board[p][r][c] == val &&
				   board[p][r + 1][c - 1] == val &&
				   board[p][r + 2][c - 2] == val &&
				   board[p][r + 3][c - 3] == val {
					return {
						found = true,
						player = p + 1,
						coords = {{r, c}, {r + 1, c - 1}, {r + 2, c - 2}, {r + 3, c - 3}},
					}
				}
			}
		}
	}
	return {found = false}
}

turn :: proc(board: helpers.gameboard, network: ^helpers.NETWORK_WEIGHTS, player: int) -> bool {
	return true
}

print_board :: proc(board: helpers.gameboard, win_state: helpers.Win_State) {
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
			color := helpers.ANSI_RESET

			if board[0][r][c] == 1 {
				symbol = "O"
				color = is_winning_piece ? helpers.ANSI_GREEN : helpers.ANSI_RED
			} else if board[1][r][c] == 1 {
				symbol = "X"
				color = is_winning_piece ? helpers.ANSI_GREEN : helpers.ANSI_YELLOW
			}

			fmt.printf("%s%s%s ", color, symbol, helpers.ANSI_RESET)
		}
		fmt.println("|")
	}
	fmt.println("---------------")
}

init_game :: proc() -> helpers.Training_Sample {
	gameboard: helpers.gameboard = {}

	sample: helpers.Training_Sample = {
		board  = gameboard,
		z_true = 0,
	}

	return sample
}


play_game :: proc(
	champion: ^helpers.NETWORK_WEIGHTS,
	challenger: ^helpers.NETWORK_WEIGHTS,
	first_player: int,
) {
	player_turn := first_player
	data: helpers.Training_Sample = {}
	game: for _ in 0 ..< 42 {
		if player_turn == 0 {
			turn(data.board, champion, 0)
			player_turn = 1
		} else {
			turn(data.board, challenger, 1)
			player_turn = 0
		}
		data.state = check_win(data.board)
		if data.state.found {
			break game
		}

	}
}
