# The Doorman Game

A conversational social engineering simulator featuring a dual-agent architecture where users attempt to persuade an AI Doorman to gain entry to an exclusive Dubai nightclub.

## Architecture

- **Agent A: The Doorman** - Front-facing conversational interface with rich persona
- **Agent B: The Judge** - Hidden scoring system that evaluates persuasiveness
- **State Manager** - Tracks cumulative influence meter (win at 100+)

## Tech Stack

- **Backend:** Python + FastAPI
- **Frontend:** React + Vite
- **LLM:** OpenRouter API (Meta Llama models via `chat/completions`)

## Quick Setup

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Copy env.example to .env and add your OpenRouter API key
cp env.example .env
# Edit .env and add your OPENROUTER_API_KEY

uvicorn main:app --reload
```

Backend runs on: `http://localhost:8000`

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: `http://localhost:3000`

### 3. Play!

Open `http://localhost:3000` in your browser and start chatting!

## Game Flow

1. User sends message to Doorman
2. Judge evaluates message and assigns score (-20 to +20)
3. Influence meter updates
4. Doorman responds based on conversation history
5. Win condition: Score reaches 100+

## Project Structure

```
.
├── backend/          # FastAPI server
├── frontend/        # React app
├── README.md
└── requirements.md
```

