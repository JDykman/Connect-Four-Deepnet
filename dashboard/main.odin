package dashboard

import "core:fmt"
import "core:math"
import rl "vendor:raylib"

// Adjust this import to point to your actual brain/engine package
import brain "../brain"

// Screen dimensions
WINDOW_WIDTH :: 1200
WINDOW_HEIGHT :: 800

// A helper to turn a float weight (e.g., -0.1 to 0.1) into a visible color.
// Negative = Red, Positive = Green. The closer to 0, the darker it is.
weight_to_color :: proc(weight: f32, max_val: f32 = 0.15) -> rl.Color {
	// Normalize the weight between -1.0 and 1.0 based on an expected max
	normalized := math.clamp(weight / max_val, -1.0, 1.0)

	intensity := u8(math.abs(normalized) * 255.0)

	if normalized > 0 {
		return rl.Color{0, intensity, 0, 255} // Green for positive
	} else {
		return rl.Color{intensity, 0, 0, 255} // Red for negative
	}
}

// Renders the 16 Conv filters (both Player 1 and Player 2 channels)
draw_conv_weights :: proc(weights: ^brain.CONV_WEIGHTS, start_x: i32, start_y: i32) {
	rl.DrawText("CONV_WEIGHTS (16 Filters, 2 Channels)", start_x, start_y - 30, 20, rl.LIGHTGRAY)

	cell_size :: 15
	padding :: 5
	filter_spacing :: 20

	for f in 0 ..< 16 {
		// Lay them out in a 4x4 grid
		grid_x := i32(f % 4)
		grid_y := i32(f / 4)

		offset_x := start_x + grid_x * (cell_size * 6 + padding + filter_spacing)
		offset_y := start_y + grid_y * (cell_size * 3 + filter_spacing + 20)

		rl.DrawText(rl.TextFormat("F%d", f), offset_x, offset_y - 15, 10, rl.GRAY)

		for ch in 0 ..< 2 { 	// P1 and P2 channels
			ch_offset_x := offset_x + i32(ch) * (cell_size * 3 + padding)

			for r in 0 ..< 3 {
				for c in 0 ..< 3 {
					w := weights[f][ch][r][c]
					color := weight_to_color(w, 0.2) // Conv weights have slightly higher variance

					rect_x := ch_offset_x + i32(c) * cell_size
					rect_y := offset_y + i32(r) * cell_size

					rl.DrawRectangle(rect_x, rect_y, cell_size - 1, cell_size - 1, color)
				}
			}
		}
	}
}

// Renders the 672 Dense weights as a 28x24 grid block
draw_dense_weights :: proc(weights: ^brain.DENSE_WEIGHTS, start_x: i32, start_y: i32) {
	rl.DrawText("DENSE_WEIGHTS (672 Inputs)", start_x, start_y - 30, 20, rl.LIGHTGRAY)

	cell_size :: 10
	cols :: 28

	for i in 0 ..< 672 {
		grid_x := i32(i % cols)
		grid_y := i32(i / cols)

		w := weights[0][i]
		color := weight_to_color(w, 0.05) // Dense weights are tiny, so lower the max_val

		rect_x := start_x + grid_x * cell_size
		rect_y := start_y + grid_y * cell_size

		rl.DrawRectangle(rect_x, rect_y, cell_size - 1, cell_size - 1, color)
	}
}

// ---------------------------------------------------------
// Engine Integration Procs (Call these from your main loop)
// ---------------------------------------------------------

// Call this once before your training loop starts
init_window :: proc() {
	rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Connect 4 AI - Live Training")
	rl.SetTargetFPS(60) // Limits how fast the window draws to save CPU
}

// Call this when your program is exiting
close_window :: proc() {
	rl.CloseWindow()
}

// Call this every X iterations to draw the current brain state.
// Returns 'false' if the user clicked the Window 'X' button.
render_frame :: proc(
	conv_w: ^brain.CONV_WEIGHTS,
	dense_w: ^brain.DENSE_WEIGHTS,
	current_game: int,
) -> bool {

	// Check if the user wants to quit
	if rl.WindowShouldClose() {
		return false
	}

	rl.BeginDrawing()
	rl.ClearBackground(rl.Color{15, 15, 20, 255})

	// Draw the Header
	header_text := rl.TextFormat("Live Training - Game/Epoch: %d", current_game)
	rl.DrawText(header_text, 30, 30, 30, rl.WHITE)
	rl.DrawText("Press 'ESC' or click 'X' to gracefully stop training.", 30, 65, 20, rl.GRAY)

	// Draw the Brain (using the procs we wrote earlier)
	draw_conv_weights(conv_w, 50, 150)
	draw_dense_weights(dense_w, 600, 150)

	rl.EndDrawing()

	return true
}

// ---------------------------------------------------------
// This main() ONLY runs if you execute `odin run dashboard`
// ---------------------------------------------------------
main :: proc() {
	rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Connect 4 AI - Live Brain View")
	defer rl.CloseWindow()
	rl.SetTargetFPS(60)

	// Mock the data using your exact initialization math
	fmt.println("Dashboard running in Standalone Mode. Seeding mock weights...")
	mock_conv, mock_dense := brain.seed_weights()

	// Main render loop
	for !rl.WindowShouldClose() {

		// --- Optional: Mock training by slightly nudging the weights every frame ---
		// Uncomment this later if you want to watch them "shimmer" to ensure the color map works
		/*
        for i in 0..<672 {
            mock_dense[0][i] += rl.GetRandomValue(-100, 100) / 100000.0
        }
        */

		rl.BeginDrawing()
		rl.ClearBackground(rl.Color{15, 15, 20, 255}) // Dark theme

		rl.DrawText("Neural Network Architecture Inspector", 30, 30, 30, rl.WHITE)
		rl.DrawText("Standalone Mock Mode", 30, 65, 20, rl.YELLOW)

		// Draw the visualizers side-by-side
		draw_conv_weights(&mock_conv, 50, 150)
		draw_dense_weights(&mock_dense, 600, 150)

		rl.EndDrawing()
	}
}
