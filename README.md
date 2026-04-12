HatCric Predictor 🏏⚡
HatCric is a blazing-fast, lightweight full-stack cricket analytics dashboard. It pairs a smart Python/Flask proxy backend—designed to calculate real-time win probabilities and protect strict API limits—with a sleek, dark-mode UI built using Vanilla JavaScript and Tailwind CSS.

Whether a match is days away, in the middle of a tense run chase, or already finished, HatCric dynamically adjusts its UI and mathematical models to give you the data you need.

✨ Key Features
Dynamic UI States: The dashboard automatically adapts to the match context:

⏳ Upcoming Match: Shows the toss result and a countdown state.

📈 First Innings: Calculates a live "Projected Score" based on the Current Run Rate.

🎯 Second Innings: Switches to "Target Chase" mode, showing exact runs needed.

🏁 Match Finished: Displays the final result and locks the winner.

Custom Win Probability Engine: Unlike basic scorecards, HatCric calculates live win probabilities during a run chase using a custom math engine that tracks Runs Needed vs. Current Run Rate.

Smart API Caching & Polling: Built specifically to survive strict free-tier API limits (like RapidAPI's 200/month limit). The Python backend caches results for 15 seconds, and the frontend JavaScript automatically pauses fetching when the browser tab is inactive.

Zero-Build Frontend: No Webpack, no React, no Node.js required for the client. Just pure HTML, Tailwind CSS via CDN, and Vanilla ES6 JavaScript.

🛠️ Tech Stack
Frontend (Client)

HTML5 & CSS3

Tailwind CSS (via CDN with custom dark-mode theme injection)

Vanilla JavaScript (Fetch API, DOM Manipulation)

Google Fonts & Icons (Lexend, Inter, Material Symbols)

Backend (Proxy Server)

Python 

Flask (Micro web framework)

Requests (For external API fetching)

Flask-CORS (Cross-Origin Resource Sharing)

python-dotenv (Environment variable security)

Data Provider:
Cricbuzz API 
