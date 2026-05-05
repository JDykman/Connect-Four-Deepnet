package dashboard

import "core:math"
import rl "vendor:raylib"

import brain "../brain"

WINDOW_WIDTH  :: 1200
WINDOW_HEIGHT :: 800

weight_to_color :: proc(weight: f32, max_val: f32 = 0.15) -> rl.Color {
	normalized := math.clamp(weight / max_val, -1.0, 1.0)
	intensity  := u8(math.abs(normalized) * 255.0)
	if normalized > 0 {
		return rl.Color{0, intensity, 0, 255}
	}
	return rl.Color{intensity, 0, 0, 255}
}

// 16 filters, 2 channels each — shows both channels side-by-side
draw_conv1_weights :: proc(weights: brain.L1_CONV_WEIGHTS, start_x: i32, start_y: i32) {
	rl.DrawText("CONV1", start_x, start_y - 18, 14, rl.LIGHTGRAY)

	cell_size      :: 12
	chan_gap       :: 4
	filter_spacing :: 14

	for f in 0 ..< brain.filter_count {
		grid_x   := i32(f % 4)
		grid_y   := i32(f / 4)
		offset_x := start_x + grid_x * (cell_size * 6 + chan_gap + filter_spacing)
		offset_y := start_y + grid_y * (cell_size * 3 + filter_spacing + 14)

		for ch in 0 ..< 2 {
			ch_x := offset_x + i32(ch) * (cell_size * 3 + chan_gap)
			for r in 0 ..< 3 {
				for c in 0 ..< 3 {
					color := weight_to_color(weights[f][ch][r][c], 0.2)
					rl.DrawRectangle(ch_x + i32(c) * cell_size, offset_y + i32(r) * cell_size, cell_size - 1, cell_size - 1, color)
				}
			}
		}
	}
}

// 16 filters, 16 input channels aggregated to mean-abs per spatial position
draw_lx_conv_weights :: proc(weights: brain.LX_CONV_WEIGHTS, label: cstring, start_x: i32, start_y: i32) {
	rl.DrawText(label, start_x, start_y - 18, 14, rl.LIGHTGRAY)

	cell_size      :: 12
	filter_spacing :: 14

	for f in 0 ..< brain.filter_count {
		grid_x   := i32(f % 4)
		grid_y   := i32(f / 4)
		offset_x := start_x + grid_x * (cell_size * 3 + filter_spacing)
		offset_y := start_y + grid_y * (cell_size * 3 + filter_spacing + 14)

		for kr in 0 ..< 3 {
			for kc in 0 ..< 3 {
				sum: f32 = 0
				for ch in 0 ..< brain.filter_count {
					sum += abs(weights[f][ch][kr][kc])
				}
				color := weight_to_color(sum / f32(brain.filter_count), 0.15)
				rl.DrawRectangle(offset_x + i32(kc) * cell_size, offset_y + i32(kr) * cell_size, cell_size - 1, cell_size - 1, color)
			}
		}
	}
}

draw_dense_weights :: proc(weights: brain.DENSE_WEIGHTS, start_x: i32, start_y: i32) {
	rl.DrawText("DENSE", start_x, start_y - 18, 14, rl.LIGHTGRAY)

	cell_size :: 10
	cols      :: 28

	for i in 0 ..< 672 {
		color := weight_to_color(weights[0][i], 0.05)
		rl.DrawRectangle(start_x + i32(i % cols) * cell_size, start_y + i32(i / cols) * cell_size, cell_size - 1, cell_size - 1, color)
	}
}

init_window :: proc() {
	rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Connect 4 AI - Live Training")
	rl.SetTargetFPS(30)
}

close_window :: proc() {
	rl.CloseWindow()
}

render_frame :: proc(weights: brain.NETWORK_WEIGHTS, current_game: int) -> bool {
	if rl.WindowShouldClose() {
		return false
	}

	rl.BeginDrawing()
	rl.ClearBackground(rl.Color{15, 15, 20, 255})

	rl.DrawText(rl.TextFormat("Game %d", current_game), 20, 18, 20, rl.WHITE)

	draw_conv1_weights(weights.conv1, 20, 100)
	draw_lx_conv_weights(weights.conv2, "CONV2", 430, 100)
	draw_lx_conv_weights(weights.conv3, "CONV3", 710, 100)
	draw_dense_weights(weights.dense, 20, 430)

	rl.EndDrawing()
	return true
}

main :: proc() {
	rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Connect 4 AI - Live Brain View")
	defer rl.CloseWindow()
	rl.SetTargetFPS(30)

	for !rl.WindowShouldClose() {
		rl.BeginDrawing()
		rl.ClearBackground(rl.Color{15, 15, 20, 255})
		rl.DrawText("Standalone Mode", 20, 20, 20, rl.YELLOW)
		rl.EndDrawing()
	}
}
