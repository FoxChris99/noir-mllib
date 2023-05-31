use std::cmp::Ordering;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::ops::{AddAssign, Div};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use noir::prelude::*;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Default, Serialize, Deserialize, Clone, Debug)]
struct Point {
    coords: Vec<f64>
}

impl Point {
    fn distance_to(&self, other: &Point) -> f64 {
        self.coords.iter().zip(other.coords.iter())
        .map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>()
        .sqrt()
    }
}

impl AddAssign for Point {
    fn add_assign(&mut self, other: Point) {
        self.coords.iter_mut().zip(other.coords.iter()).for_each(|(a,b)| *a+=b);
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        let precision = 0.001;
        let mut res = true;
        self.coords.iter().zip(other.coords.iter()).for_each(|(a,b)| if (a-b).abs() > precision {res = false;});
        res
    }
}

impl Eq for Point {}

impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.coords.partial_cmp(&other.coords) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.coords.partial_cmp(&other.coords)
    }
}

impl Ord for Point {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for el in self.coords.iter(){
            el.to_le_bytes().hash(state);
        }    
    }
}

impl Div<f64> for Point {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        let mut new_coords = self.coords.clone();
        new_coords.iter_mut().for_each(|a| *a/=rhs);
        Self {
            coords: new_coords
        }
    }
}

//Take first n points in csv file as the starting centroids
fn read_centroids(filename: &str, n: usize) -> Vec<Point> {
    let file = File::open(filename).unwrap();
    csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file)
        .into_deserialize::<Point>()
        .map(Result::unwrap)
        .take(n)
        .collect()
}

//Find the Point in a vector which has the lowest distance from another specified Point
fn select_nearest(point: &Point, old_centroids: &Vec<Point>) -> Point {
    let res: Point;
    res = old_centroids
        .iter()
        .map(|c| (c, c.distance_to(&point)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0.clone();
    res
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct State {
    iter_count: i64,
    old_centroids: Vec<Point>,
    centroids: Vec<Point>,
}

impl State {
    fn new(old_centroids: Vec<Point>) -> State {
        State {
            old_centroids,
            ..Default::default()
        }
    }
}

fn main() {
    let (config, args) = EnvironmentConfig::from_args();
    if args.len() != 3 {
        panic!("Pass the number of centroid, the number of iterations and the dataset path as arguments");
    }
    let num_centroids: usize = args[0].parse().expect("Invalid number of centroids");
    let num_iters: usize = args[1].parse().expect("Invalid number of iterations");
    let path = &args[2];

    let mut env = StreamEnvironment::new(config);

    env.spawn_remote_workers();

    let centroids = read_centroids(path, num_centroids);
    assert_eq!(centroids.len(), num_centroids);
    let initial_state = State::new(centroids);

    let source = CsvSource::<Point>::new(path).has_headers(false);
    let res = env
        .stream(source)
        .replay(
            num_iters,
            initial_state,
            |s, state| {
                    s.map(move |point| (point.clone(), select_nearest(&point, &state.get().old_centroids)))
                    .group_by_avg(|(_p, c)| c.clone(), |(p, _c)| p.clone())
                    .drop_key()
            },
            |update: &mut Vec<Point>, p| update.push(p),
            move |state, mut update| {                
                state.centroids.append(&mut update);
            },
            |state| {
                let mut changed: bool = false;
                state.iter_count += 1;
                state.centroids.sort_unstable();
                state.old_centroids.sort_unstable();
                if state.centroids != state.old_centroids {
                    changed = true;
                    state.old_centroids.clear();
                    state.old_centroids.append(&mut state.centroids);
                } 
                changed
            },
        )
        .collect_vec();

    let start = Instant::now();
    env.execute();
    let elapsed = start.elapsed();
    if let Some(res) = res.get() {
        let state = &res[0];
        eprintln!("Iterations: {}/{}", state.iter_count,num_iters);
        eprintln!("Centroids: {:?}", state.centroids.len());
        eprintln!("Old Centroids: {:?}", state.old_centroids.len());
    }
    eprintln!("Elapsed: {elapsed:?}");
}
