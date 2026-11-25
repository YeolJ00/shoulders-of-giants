"""lsim -- a learned household simulator.

One generative model over explicit 3D meshes (rigid objects, hinged parts, cloth). Every input is
a slot that is given or empty; the model generates the empty ones. Pretrained by distilling a
classical simulator (Newton), fine-tuned on real motion. Design and plan: docs/DIRECTION.md.
"""
