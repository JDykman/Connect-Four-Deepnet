package main

import "core:fmt"
import "core:os"
import "game"


main :: proc() {
	gen_flag := false
	train_flag := false

	if len(os.args) <= 1 {
		fmt.println("No args")
	}

	for arg, i in os.args {
		switch arg {
		case "gen":
			gen_flag = true
		case "train":
			train_flag = true
		}
	}

	if gen_flag {
		dataset_file :: "connect4_training_data.bin"
		total_games_to_generate :: 10_000

		// Gen Step
		fmt.printfln("Generating %v games...", total_games_to_generate)
		for i in 1 ..= total_games_to_generate {
			board, win_state := game.play_game()

			sample := game.Training_Sample {
				board = board,
				state = win_state,
			}

			success := game.dump_dataset(dataset_file, []game.Training_Sample{sample})

			if !success {
				fmt.println("Fatal Error writing to disk. Halting generation.")
				break
			}

			if i % 1000 == 0 {
				fmt.printfln("Generated %v games...", i)
			}
		}

		fmt.println("Dataset generation complete!")
	}
}
