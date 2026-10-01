# A fly that has to stay alive

The MaleCNS connectome wired up as a spiking brain, given a body with a metabolism,
and dropped into an arena with five request pads. It has to find food and water or
it dies.

```sh
./.venv/Scripts/python.exe sim/run.py                  # 3D viewer on :5050
./.venv/Scripts/python.exe sim/run.py --headless 300   # run 300 sim-seconds, print a report
./.venv/Scripts/python.exe sim/probe.py               # characterise sensor -> motor transfer
./.venv/Scripts/python.exe sim/calibrate.py           # re-derive the firing-rate calibration
```

`sim/build_brain.py` must run once first; it compiles the DuckDB connectome into
`sim/brain_malecns.npz` (75 MB).

## What is actually real

This matters more than anything else here, so it is listed explicitly.

**Real data, straight from MaleCNS v1.0:**

- 166,700 neurons and all 25,582,938 neuron-to-neuron connections, at their real
  synapse counts.
- The sign of every connection, from the presynaptic cell's predicted transmitter
  (ACH excitatory; GABA, glutamate and histamine inhibitory).
- The identity of every sensory population that can be stimulated: gustatory
  receptor neurons by bristle class (labellar, leg, wing, taste peg, pharyngeal),
  hygrosensory, thermosensory, olfactory, visual, and three mechanosensory classes.
- The identity of every motor pool that is read out: 67 proboscis muscle motor
  neurons, leg motor neurons by segment and side, wing, abdominal, neck.
- The mushroom body: 4,064 Kenyon cells, 340 dopaminergic neurons, 97 output
  neurons, and the 61,210 KC→MBON synapses that plasticity acts on.

**Model, i.e. invented by me:**

- Leaky integrate-and-fire cell dynamics. Parameters follow the published
  whole-brain LIF model of the FlyWire connectome (Shiu et al., *Nature* 2024):
  20 ms membrane time constant, 2.2 ms refractory period, 7 mV threshold above
  rest, 0.275 mV per synapse.
- A global synaptic gain of 0.08 and an extra ×1.4 on inhibition. **These are
  tuned.** At raw published weights this graph is bistable: either silent, or it
  ignites and pins at ~30 Hz. The gain was swept until the resting population rate
  sat near 1.8 Hz, which is what fly neurons actually do. See `sim/calibrate.py`.
- Tonic drive and membrane noise. A connectome on its own is silent; real brains
  are spontaneously active.
- The entire metabolism: energy, crop, water, sodium, damage, repair, growth,
  death thresholds, and the per-spike energy cost.
- Arena physics, and the muscle model that turns motor-neuron firing rates into
  velocity and yaw.
- A spontaneous foraging drive, scaled by hunger. Baseline leg-motor firing means
  "standing", so a purely readout-driven fly never leaves the spot it starts on.
  Real flies walk spontaneously and walk more when deprived, driven by central
  pattern generators this static wiring diagram does not contain.

## The pads

Five pads around the arena. Each has:

- **a cue** — a fixed sparse subset of the 2,639 real olfactory receptor neurons.
  That is what an odour physically is to a fly: a sparse ORN activation pattern.
  The fly smells a pad's cue whenever it is nearby.
- **a payload** — dispensed after the fly dwells on the pad for 600 ms. Delivery
  drives a real sensory population.

| Pad | Payload | Sensory population driven |
|---|---|---|
| SUGAR | sucrose → crop → energy | taste-peg GRNs |
| WATER | water → hydration | hygrosensory neurons (HRN_VP1d/VP4/VP5) |
| SALT | sodium | leg-bristle GRNs |
| BITTER | costs energy, does damage | pharyngeal GRNs |
| BLANK | nothing | — |

**I did not choose which GRN class means food.** MaleCNS types gustatory neurons
anatomically and does not record what they taste. So `sim/probe.py` drove each
class in turn and measured the 67 proboscis motor neurons. Result:

```
grn_tastepeg     +22.6 Hz   <- the only class that excites feeding
grn_labellar      -9.2 Hz
grn_pharyngeal    -8.0 Hz   <- suppresses feeding at every amplitude tested
```

Taste peg became sugar and pharyngeal became bitter because that is what the
wiring does. The appetitive/aversive opponency was recovered from the connectome,
not imposed on it.

The same probe found other things worth knowing, all consistent with real fly
biology:

```
mechano_proprio  -> wing +29.5 Hz, front legs +10.4 Hz     (locomotion)
mechano_tactile  -> all six leg pools +5 to +12 Hz         (escape)
mechano_head     -> front legs +19.5/+18.9 Hz              (grooming)
olfactory        -> Kenyon +79.8, DAN +116.4, MBON +144.0   (the mushroom body)
```

That last line is the one that makes the learning experiment plausible: odour
drives the mushroom body hard, which is exactly its job.

## Ingestion is not free

The fly only swallows if its proboscis motor neurons are actually firing at least
8 Hz above their own resting rate. Standing on a dispensing pad is not enough —
the network has to produce the motor command. That is the point of the closed
loop.

## Learning

KC→MBON synapses are depressed when a Kenyon cell was recently active while
dopaminergic neurons were firing. That is the established direction of the effect
in *Drosophila*, and it is the mechanism by which a fly learns that a cue predicts
a reward. Reinforcement is injected into the real DAN population, scaled by how
much the payload relieved the need it maps to.

Simplified: real mushroom-body plasticity is compartment-specific, with particular
DANs gating particular MBONs. Here the dopaminergic gate is global. So the model
can learn "this cue was recently paired with reward" but not the full
compartmental structure of valence.

## Why it runs at roughly a third of realtime

Propagation is event driven — only the outgoing synapses of neurons that actually
spiked on a given millisecond are touched. A dense matrix-vector product over
25.6M synapses costs 18.5 ms per step; the event-driven version costs about 3 ms
at resting firing rates. The viewer reports the live realtime factor, and it is
never faked to keep up: simulated time is simulated time.

## Honest limits

- **Neuromodulation is missing.** Hunger in a real fly reshapes circuit gain
  through dopamine, octopamine and neuropeptides acting on specific cells. Here it
  is a scalar on sensory gain and on the foraging drive.
- **No compartmental neurons, no synaptic delays, no short-term plasticity, no
  gap junctions.** Every cell is a point.
- **The connectome is one animal, fixed.** Real brains change their wiring; this
  one cannot, apart from the KC→MBON weights.
- **Behaviour is not validated.** The probe results above are checks on
  sensorimotor transfer, not evidence that this fly behaves like a real fly. It
  should not be read as a prediction about *Drosophila*.
