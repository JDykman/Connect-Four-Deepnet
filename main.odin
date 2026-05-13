package main

import "brain"
import "core:fmt"
import "core:os"
import "core:strconv"
import "dashboard"
import "helpers"

//TODO multithread training?
main :: proc() {
	train_flag := false
	run_count := 0


	if len(os.args) <= 1 {
		fmt.println("No args")
	}

	for arg in os.args[1:] {
		if arg == "train" {
			train_flag = true
		}
		if n, ok := strconv.parse_int(arg); ok {
			run_count = n
		}
	}

	if train_flag {
		dashboard.init_window()
		defer dashboard.close_window()

		network := brain.seed_weights(context.temp_allocator)
		step := 0

		render_every :: 100
		fmt.printfln(
			"%sRunning %v training loops%s",
			helpers.ANSI_YELLOW,
			run_count,
			helpers.ANSI_RESET,
		)

		outer: for _ in 0 ..< run_count {
		}
	}
}
