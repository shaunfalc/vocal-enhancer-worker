# VocalRecipe — Worker Spec (MOVED)

⚠️ **The VocalRecipe worker is a SEPARATE repo, not part of this worker.**

See: `shaunfalc/vocal-recipe-worker`

Reasons for separation:
- VocalRecipe uses Librosa (CPU only) — no CUDA/GPU needed
- VocalEnhancer uses Resemble Enhance (requires GPU)
- Different scaling, different cost, independent deployments
- Keeping them separate prevents dependency conflicts and image bloat

The full technical spec is at:
`vocal-platform/marketplace/docs/PRD-vocalrecipe.md`
