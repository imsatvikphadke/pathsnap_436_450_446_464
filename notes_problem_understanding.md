# PathSnap - Team 5: Problem Understanding & Architecture Overview

**Author:** Sanyam Deven Hiran (PES2UG24CS446)  
**Team 5:** Samyak Sanklecha (436), Satvik Phadke (450), Sanyam Deven Hiran (446), Shaurya Ishan (464)  
**Course:** UE24CS352A - Machine Learning | PES University  

---

## 1. Challenge & Problem Definition
GPS tracking data collected in urban environments is severely degraded by multipath propagation, satellite occlusion, high-rise buildings, clock drift, and sampling dropouts. Consequently, raw GPS observations exhibit noisy deviations, cross onto parallel streets, or display discontinuous positional jumps.

**Map-matching** is the task of inferring the true, physically traversed sequence of road segments (arcs) from a sequence of noisy GPS trajectory points. In **PathSnap**, the goal is to develop an end-to-end, distributed machine learning system that takes raw GPS trajectories and a localized road network graph and reconstructs a valid, 100% drivable trajectory across all evaluation trips.

---

## 2. Dataset Structure & Characteristics
The dataset is structured on a per-trip basis across 80 training trips, 10 student-testing trips, and 10 held-out evaluator trips. Each trip folder consists of 4 distinct files:
1. **`<trip_id>.track`**: Raw GPS trajectory observations containing `longitude`, `latitude`, and `timestamp` (seconds). The sampling frequency is nominally 1 Hz (~97% of intervals are 1 s, with irregular gaps up to 300+ s).
2. **`<trip_id>.nodes`**: Road network vertices with 0-based indices representing `longitude` and `latitude`.
3. **`<trip_id>.arcs`**: Directed road network edges with 0-based indices representing `from_node_id` and `to_node_id`. Directed arcs distinguish opposite lanes of the same physical road.
4. **`<trip_id>.route`**: Ground-truth chronological sequence of traversed `arc_id`s, forming a continuous directed path (`to_node` of arc $k$ equals `from_node` of arc $k+1$).

### Key Geographical & Preprocessing Considerations:
- **Global Diversity & Local Projection**: Trips are located across diverse global cities (Europe, North/South America, Asia, Africa). Euclidean distance in degrees is geographically invalid, and a single global UTM projection cannot be applied. Each trip is transformed into a local Cartesian tangent plane (metres) centered on its initial GPS coordinate $(lon_0, lat_0)$.
- **Directed Arc Discrimination**: Dual-direction roads share identical coordinates with opposite headings. Pure distance-based matching confuses opposite traffic lanes; therefore, trajectory velocity vectors, heading alignments, and temporal transitions are mandatory.
- **Sequential Route vs. Per-Point Ground Truth**: The raw route is an ordered sequence of arcs, not an explicit per-GPS-point assignment. Preprocessing must derive ground-truth arc correspondences by projecting GPS points onto the known route trajectory.

---

## 3. End-to-End System Pipeline & Architecture
The system architecture decouples feature extraction, discriminative candidate scoring, sequential topological decoding, and networked inference into clean, reproducible stages:

```
[Raw Trip Data: .track, .nodes, .arcs]
                │
                ▼ (Samyak)
       [Data Preprocessing & Cleaning]
       • Local metre projection (x, y)
       • Speed, heading, gap & jump detection
                │
                ▼ (Samyak v1 / Satvik v2)
       [Spatial Candidate Generation]
       • KD-Tree spatial index over arc midpoints/segments (Radius R=50m, Top-K=10)
       • Feature extraction (distance, heading difference, speed, network degree)
                │
                ▼ (Sanyam)
       [Feature Selection (Train-Only)]
       • Collinearity filtering (|r| > 0.95)
       • Mutual Information (MI) ranking
       • Gradient Boosted Feature Importance
                │
                ▼ (Ishan)
       [Candidate Classification Model]
       • 5-6 ML algorithms evaluated (GroupKFold by trip_id)
       • Bayesian Hyperparameter Optimization (Optuna)
       • P(candidate = true_arc | features)
                │
                ▼ (Sanyam)
       [HMM & Viterbi Sequential Decoder]
       • Hidden States: Candidate road arcs at time t
       • Emission Probabilities: Re-normalized classifier probability log-scores
       • Transition Probabilities: Topological connectivity & graph distance penalty
       • Viterbi Trellis Backtracking (O(T · K²))
                │
                ▼ (Sanyam Backend & Satvik Frontend)
    [Two-Laptop Distributed Deployment]
    • Laptop 1: FastAPI Server hosting model, graph cache & HMM engine
    • Laptop 2: Streamlit Interactive Client with Folium Map Visualizations
```

---

## 4. Evaluation Metrics & Integrity
To rigorously evaluate individual components and the global trajectory:
1. **Observation-Level Top-1 Accuracy**: Fraction of GPS points where the top-scoring candidate matches the reference arc.
2. **Candidate Miss Rate**: Fraction of GPS points where the ground-truth arc is absent from the top-$K$ candidate set (establishes the theoretical upper bound on accuracy).
3. **Route Continuity**: Proportion of consecutive arc transitions in the predicted route where `to_node(arc[t]) == from_node(arc[t+1])`. Evaluates drivability.
4. **Sequence-Level F1**: Precision, recall, and F1 between the set of unique predicted route arcs and true route arcs.
5. **Macro F1 & Binary Log Loss**: Evaluates candidate classifier calibration and discrimination on the binary candidate task.
6. **Inference Latency**: Measured in wall-clock milliseconds per trip.

**Academic Integrity & Zero-Leakage Guarantee**:
- No evaluator-test trip data is accessed during training, feature selection, or hyperparameter optimization.
- Ground truth reference routes remain strictly server-side on Laptop 1 during final evaluation and are never transferred to the UI client.
