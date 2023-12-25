use std::fs::File;
use std::io::Read;
use rand::{self, Rng};
mod constants;
mod mst;
use std::collections::BinaryHeap;
use constants::*;
use image::{GenericImageView, ImageBuffer, RgbImage, imageops};



use rayon::prelude::*;
use crate::mst::State;

fn upscale_letter(letter: [u8; LETTER_BUF]) -> [bool; MAZE_BUF] {
    let mut out_buf = [false; MAZE_BUF];
    assert!(MAZE_SIZE % LETTER_SIZE == 0);

    for y in 0..LETTER_SIZE {
        for x in 0..LETTER_SIZE {
            let letter_content = letter[y * LETTER_SIZE + x] == 0;
            for yi in 0..RATIO {
                for xi in 0..RATIO {
                    out_buf[(y * RATIO + yi) * MAZE_SIZE + (x * RATIO + xi)] = letter_content;
                }
            }
        }
    }
    out_buf
}

fn p(c: (usize, usize)) -> usize {
    c.1 * MAZE_SIZE + c.0
}

fn pi(c: (i32, i32)) -> usize {
    c.1 as usize * MAZE_SIZE + c.0 as usize
}


fn on_map(x: usize) -> bool {
    x >= 0 && x < MAZE_SIZE
}
fn c_on_map(x: (usize, usize)) -> bool {
    on_map(x.0) && on_map(x.1)
}
fn on_map_i(x: i32) -> bool {
    x >= 0 && x < (MAZE_SIZE-1) as i32
}
fn c_on_mapi(x: (i32, i32)) -> bool {
    on_map_i(x.0) && on_map_i(x.1)
}

fn solve_maze(walls: &[bool; MAZE_BUF]) -> Option<(i32, [bool; MAZE_BUF])> {
    let mut visited_distance = [u32::MAX; MAZE_BUF];
    let start_coord = (RATIO * 2 + 1, 1 as usize);
    let end_coord = (RATIO * 10 - 1, MAZE_SIZE - 3);
    if walls[p(start_coord)] || walls[p(end_coord)] {
        return None;
    }
    let coord_offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)];

    visited_distance[p(start_coord)] = 0;

    let mut prev_coords: Vec<(usize, usize)> = Vec::new();
    prev_coords.push(start_coord);

    while prev_coords.len() > 0 {
        let mut cur_coords: Vec<(usize, usize)> = Vec::new();
        for (px, py) in prev_coords {
            for (xi, yi) in coord_offsets.iter(){
                let nc = ((px as i32 + xi) as usize, (py as i32 + yi) as usize);
                if c_on_map(nc) && !walls[p(nc)] && visited_distance[p(nc)] == u32::MAX {
                    visited_distance[p(nc)] = visited_distance[p((px, py))] + 1;
                    cur_coords.push(nc);
                }
            }
        
        }
        prev_coords = cur_coords;
    }
    if visited_distance[p(end_coord)] == u32::MAX {
        return None;
    }
    let mut visit_map = [false; MAZE_BUF];
    let mut cur_coords = vec![end_coord];
    let mut cur_dist = visited_distance[p(end_coord)];
    let mut total_ambiguity: i32 = -(cur_dist as i32);
    while cur_coords[0] != start_coord {
        let mut next_coords: Vec<(usize, usize)> = Vec::new();
        for (px, py) in cur_coords {
            for (xi, yi) in coord_offsets.iter(){
                let nc = ((px as i32 + xi) as usize, (py as i32 + yi) as usize);
                if !visit_map[p(nc)] && visited_distance[p(nc)] == cur_dist - 1 {
                    visit_map[p(nc)] = true;
                    total_ambiguity += 1;
                    next_coords.push(nc);
                }
            }
        }
        cur_dist -= 1;
        cur_coords = next_coords;
    }

    Some((total_ambiguity, visit_map))
}

const GEN_SIZE: i32 = (MAZE_SIZE / 2) as i32 - 1;

fn print_walls(walls: &[bool; MAZE_BUF]) {
    let start_coord = (RATIO * 2 + 1, 1 as usize);
    let end_coord = (RATIO * 10 - 1, MAZE_SIZE - 3);
    for y in 0..MAZE_SIZE {
        for x in 0..MAZE_SIZE {
            let is_end = (x, y) == start_coord || (x, y) == end_coord;
            let char = if walls[y * MAZE_SIZE + x] {
                '#'
            } else {
                if is_end {
                    'o'
                } else {
                    '.'
                }
            };
            print!("{}", char);
        }
        println!("");
    }
}

fn print_walls_path(walls: &[bool; MAZE_BUF]) {
    let start_coord = (RATIO * 2 + 1, 1 as usize);
    let end_coord = (RATIO * 10 - 1, MAZE_SIZE - 3);
    let sol = solve_maze(&walls).unwrap().1;
    for y in 0..MAZE_SIZE {
        for x in 0..MAZE_SIZE {
            let is_end = sol[p((x, y))];
            let char = if walls[y * MAZE_SIZE + x] {
                '#'
            } else {
                if is_end {
                    'o'
                } else {
                    '.'
                }
            };
            print!("{}", char);
        }
        println!("");
    }
}

fn gen_rand_maze() -> [bool; MAZE_BUF] {
   gen_maze(&gen_maze_weights())
}

fn eval_maze(letter_pattern: &[bool; MAZE_BUF], walls: &[bool; MAZE_BUF]) -> i32 {
    let solution = solve_maze(walls);
    match solution {
        None => i32::MIN,
        Some((amb, solution_visits)) => solution_visits
            .iter()
            .zip(letter_pattern.iter())
            .map(|(vis, pat)| if *vis == *pat { 1 } else { -1 })
            .sum(),
    }
}

fn gen_maze_weights()->Box<[i32; MAZE_BUF]>{
    let mut rng = rand::thread_rng();
    let weights = [0 as i32; MAZE_BUF].map(|_|{
        let cur_weight: i16 = rng.gen();
        cur_weight as i32
    });
    Box::new(weights)
}

fn gen_maze(weights: &[i32; MAZE_BUF])-> [bool; MAZE_BUF]{
    let start_pos = State{cost:0,position:(1,1), offset:(0,0)};
    
    let mut positions:BinaryHeap<State> = BinaryHeap::new();
    positions.push(start_pos);

    let mut walls = [true; MAZE_BUF];

    while positions.len() > 0{
        let top = positions.pop().unwrap();
        if !walls[pi(top.position)]{
            continue;
        }
        walls[pi(top.position)] = false;
        let (px, py) = top.position;
        let (ox, oy) = top.offset;
        let prev_wall = (px - ox as i32, py - oy as i32);

        walls[pi(prev_wall)] = false;
        let coord_offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)];

        for (ox, oy)  in coord_offsets{
            let next_p = (px + ox*2 as i32, py + oy*2 as i32);
            let next_wall = (px + ox as i32, py + oy as i32);
            if c_on_mapi(next_p) && walls[pi(next_p)]{
                positions.push(State{
                    cost: top.cost + weights[pi(next_wall)],
                    offset: (ox as i16, oy as i16),
                    position: next_p,
                });
            }
        }
    }
    walls
}

fn mutate_weights(weights: &[i32; MAZE_BUF])->[i32; MAZE_BUF]{
    let mut weights_copy = weights.clone();
    let mut rng = rand::thread_rng();
    let start_x = rng.gen_range(0..MAZE_SIZE-2);
    let end_x = rng.gen_range(start_x+1..MAZE_SIZE);
    let start_y = rng.gen_range(0..MAZE_SIZE-2);
    let end_y = rng.gen_range(start_y+1..MAZE_SIZE);
    for y in start_y..end_y{
        for x in start_x..end_x{
            let w:i16 = rng.gen();
            weights_copy[y*MAZE_SIZE+x] = w as i32;
        }
    }
    weights_copy
}

const NUM_GENERATIONS: usize = 2000;
const NUM_ITEMS:usize = 96;
fn generate_maze(letter_pattern: &[bool; MAZE_BUF]) -> Box<[bool; MAZE_BUF]> {
    let mut cur_weights:Vec<(i32,Box<[i32; MAZE_BUF]>)> = (0..NUM_ITEMS).map(|x|{
        let maze_weights = gen_maze_weights();
        let maze = gen_maze(&maze_weights);
        let score = eval_maze(letter_pattern, &maze);
        (score, maze_weights)
    }).collect();
    for _i in 0..NUM_GENERATIONS{
        cur_weights = cur_weights.par_iter().map(|x|{
            let new_weights = mutate_weights(&x.1);
            let new_maze = gen_maze(&new_weights);
            let new_score = eval_maze(letter_pattern, &new_maze);
            if new_score > x.0{
            (new_score, Box::new(new_weights))
            }else{
                (x.0,x.1.clone())
            }
        }).collect();
    }
    let best_weight = cur_weights.iter().reduce(|v1, v2|{
        if v1.0 > v2.0{
            v1
        }else{
            v2
        }
    }).unwrap();
    
    println!("{}",best_weight.0);
    let best_weights = best_weight.1.clone();
    let best_maze = gen_maze(&best_weights);
    Box::new(best_maze)
}

fn save_maze(walls: &[bool; MAZE_BUF], out_fname:&str) {
    let start_coord = (RATIO as u32 * 2 + 1, 1 as u32);
    let end_coord = (RATIO as u32 * 10 - 1, MAZE_SIZE as u32 - 3);

    const PIXEL_SIZE:u32 = 12;
    const IMG_SIZE:u32 = PIXEL_SIZE * MAZE_SIZE as u32;
    let mut imgbuf = image::ImageBuffer::new(MAZE_SIZE as u32, MAZE_SIZE as u32);
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let is_end = (x,y) == (start_coord) || (x,y) == end_coord;
        let val = walls[pi((x as i32,y as i32))];
        let rgb_val:[u8;3] = if is_end { [0,0,255] } else { if val {[144,238,144]} else {[255,255,255]} };
        *pixel = image::Rgb(rgb_val);
    }
    let out_img = image::imageops::resize(&imgbuf, IMG_SIZE, IMG_SIZE,image::imageops::FilterType::Nearest);
    // imgbuf.resize
    out_img.save(out_fname).unwrap();
}


fn save_maze_solved(walls: &[bool; MAZE_BUF], out_fname:&str) {
    let start_coord = (RATIO as u32 * 2 + 1, 1 as u32);
    let end_coord = (RATIO as u32 * 10 - 1, MAZE_SIZE as u32 - 3);

    let start_coord = (RATIO * 2 + 1, 1 as usize);
    let end_coord = (RATIO * 10 - 1, MAZE_SIZE - 3);
    let sol = solve_maze(&walls).unwrap().1;
    // for y in 0..MAZE_SIZE {
    //     for x in 0..MAZE_SIZE {
    const PIXEL_SIZE:u32 = 12;
    const IMG_SIZE:u32 = PIXEL_SIZE * MAZE_SIZE as u32;
    let mut imgbuf = image::ImageBuffer::new(MAZE_SIZE as u32, MAZE_SIZE as u32);
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        // let is_end = (x,y) == (start_coord) || (x,y) == end_coord;
        let is_end = sol[p((x as usize, y as usize))];
        let val = walls[pi((x as i32,y as i32))];
        let rgb_val:[u8;3] = if is_end { [0,0,255] } else { if val {[144,238,144]} else {[255,255,255]} };
        *pixel = image::Rgb(rgb_val);
    }
    let out_img = image::imageops::resize(&imgbuf, IMG_SIZE, IMG_SIZE,image::imageops::FilterType::Nearest);
    // imgbuf.resize
    out_img.save(out_fname).unwrap();
}

fn main() {
    let mut stdfile = File::open("letters.bin").unwrap();
    let mut letters: Vec<[u8; LETTER_BUF]> = Vec::new();
    for i in 0..26 {
        let mut letters_buf = [0 as u8; LETTER_BUF];
        stdfile.read(&mut letters_buf).unwrap();
        letters.push(letters_buf);
    }
    let clue = b"ssss".iter().map(|x|*x-b'a').enumerate().for_each(|(i,l)|{    
        print!("\nLetter {}\n\n",i+1);
        let maze = generate_maze(&upscale_letter(letters[l as usize]));
        print_walls(&maze);
        let fname = format!("letter_{}.png",i+1);
        let fname_solved = format!("letter_solution_{}.png",i+1);
        save_maze(&maze, &fname);
        save_maze_solved(&maze, &fname_solved);
    });
    
}
