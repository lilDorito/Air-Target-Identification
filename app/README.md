# app/

Frontend (late stage): an operator picks a sensor scenario (A/B/C) and the countries they face, enters an observation, and gets top-5 aircraft, roles and types with probabilities.

The app loads models from `../models/` and calls `src/country_filter.py` for masking and renormalising. It needs the local dataset for display names, so it is not runnable from the public repo alone.

_TODO: framework choice, but i guess Vercel would do it._
