package main

import brain "brain"
import "core:fmt"
import "core:os"
import "dashboard"
import "game"

main :: proc() {
	gen_flag := false
	train_flag := false


	if len(os.args) <= 1 {
		fmt.println("No args")
	}

	for arg in os.args[1:] {
		switch arg {
		case "gen":
			gen_flag = true
		case "train":
			train_flag = true
		}
	}

	if gen_flag {
		dataset_file :: "connect4_training_data.bin"
		total_games_to_generate :: 10000

		fmt.printfln("Generating %v games...", total_games_to_generate)
		fmt.printfln("----------------------------")
		for i in 1 ..= total_games_to_generate {
			board, win_state := game.play_game()

			sample := game.Training_Sample {
				board = board,
				state = win_state,
			}

			if !game.dump_dataset(dataset_file, []game.Training_Sample{sample}) {
				fmt.println("Fatal Error writing to disk. Halting generation.")
				break
			}

			if i % 1000 == 0 {
				fmt.printfln("Generated %v games...", i)
			}
		}
		fmt.printfln("----------------------------")
		fmt.println("Dataset generation complete!")
	}

	if train_flag {
		dashboard.init_window()
		defer dashboard.close_window()

		dataset_file :: "connect4_training_data.bin"

		data, ok := brain.load_dataset(dataset_file, context.temp_allocator)
		if !ok {
			fmt.println("Failed to load dataset")
			return
		}

		network := brain.seed_weights(context.temp_allocator)
		step := 0

		render_every :: 100

		outer: for {
			for &sample in data {
				brain.train_step(network, sample.board)
				step += 1
				if step % render_every == 0 {
					if !dashboard.render_frame(network^, step) {
						break outer
					}
				}
			}
		}
	}
}
