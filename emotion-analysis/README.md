# React Frontend Documentation

## Emotion Analysis Dashboard Frontend Documentation

**Version:** 1.0  
**Date:** February 25, 2025  
**Author:** Grok 3 (xAI)

---

## Overview

This document describes the React frontend for the Emotion Analysis Dashboard, a web application that allows users to input text, analyze emotions, topics, and Adorescores, and manage past analyses. The frontend uses React 18 (or 19, depending on compatibility), styled-components for CSS, and libraries like Recharts for visualizations, interacting with a FastAPI backend.

---

## System Requirements

- **Node.js:** Version 16.x or higher (recommended 18.x or 19.x).
- **npm:** Version 8.x or higher.
- **Browser:** Modern browsers (Chrome, Firefox, Edge, Safari) supporting ES6+ and React 18/19.

### Dependencies

- `react` (18.2.0 or 19.0.0)
- `react-dom` (18.2.0 or 19.0.0)
- `styled-components` (5.x)
- `axios` (1.x)
- `recharts` (2.x)

---

## Installation and Setup

### Install Dependencies

Run:

```bash
npm install
```

If using React 19, adjust to:

```bash
npm install --legacy-peer-deps
```

### Run the Application

Start the development server:

```bash
npm start
```

The app will run on [http://localhost:3000](http://localhost:3000).

### Backend Dependency

Ensure the FastAPI backend is running at [http://localhost:8000](http://localhost:8000) (see Python documentation for setup).

---

## Architecture and Components

The frontend is structured as a single-page application (SPA) with the following key components:

- **App.js:** The main component, managing state (analysisData, loading, error), rendering the dashboard layout, and integrating subcomponents.
- **AnalysisForm.js:** Handles text input and submission to the backend via `/analyze`, triggering emotion and Adorescore analysis.
- **AnalysisList.js:** Displays a list of past analyses, with edit and delete functionality via `/analyses`, `/analysis/{id}` (GET, PUT, DELETE).
- **EmotionRadarChart.js:** Visualizes emotions (high, medium, low activation) using Recharts’ radar charts, styled in white boxes with rounded corners.
- **AdorescoreDisplay.js:** Shows the overall Adorescore and primary emotion, styled similarly.
- **TopThemesList.js:** Lists top themes with scores and volumes, styled in a white box.
- **TopicHierarchy.js:** Displays topics, subtopics, and Adorescore breakdowns in a static list with a bar chart, styled consistently.

### State Management

Uses React hooks (`useState`, `useEffect`) for managing analysis data, loading states, and errors, with Axios for API calls.

### Styling

Utilizes styled-components for a modern, responsive design with white (`#fff`) content boxes, light gray (`#f0f0f0`) background, rounded corners, and shadows.

### Libraries

- **Recharts** for data visualizations (radar and bar charts).
- **Axios** for HTTP requests to the backend.

---

## Functionality

### Text Analysis

Users input text in `AnalysisForm`, triggering a POST to `/analyze`, displaying emotions, Adorescores, topics, and subtopics.

### Past Analyses Management

`AnalysisList` fetches data from `/analyses`, allowing users to view, edit, or delete analyses via PUT/DELETE endpoints.

### Visualization

- **Radar charts** show emotions by activation level (high, medium, low).
- **Bar charts** display Adorescore breakdowns for topics.
- **Lists** show themes, topics, and subtopics with scores.

### Error Handling

Uses an `ErrorBoundary` for global error catching, displaying user-friendly messages and reload options.

### Key Features

- Responsive layout for desktop/mobile.
- Interactive visualizations (hover tooltips, clickable past analyses).
- Real-time updates from backend responses.

---

## Usage

### Launching the App

Follow installation steps, ensure the backend is running, and navigate to [http://localhost:3000](http://localhost:3000).

### Analyzing Text

Enter text in the textarea, click "Analyze," and view results in radar charts, Adorescore, themes, topics, and subtopics.

### Managing Past Analyses

View the list of past analyses, click to select one for display, edit text via the input field, or delete entries.

### Interacting with Visualizations

Hover over radar/bar charts for tooltips showing details (e.g., emotion probabilities, Adorescores).

---

## Maintenance and Troubleshooting

### Common Issues

- **Backend Unavailable:** Ensure [http://localhost:8000](http://localhost:8000) is running. Check network errors in the console.
- **Dependency Conflicts:** Run `npm audit fix` if dependency issues arise, or use `--legacy-peer-deps` for React 19 compatibility.
- **Rendering Errors:** Check `ErrorBoundary` logs in the console for component-specific errors.

### Updates

- Update React, Recharts, and styled-components regularly via `npm update`.
- Sync with backend changes to maintain API compatibility.

### Testing

Use Jest and React Testing Library for unit/integration tests:

```bash
npm install --save-dev jest @testing-library/react @testing-library/jest-dom
```

Test components (e.g., `AnalysisForm`, `EmotionRadarChart`) with mock API responses.

---

## API Integration

Communicates with the FastAPI backend at [http://localhost:8000](http://localhost:8000):

- **GET** `/analyses`: Fetch past analyses.
- **GET** `/analysis/{id}`: Fetch a specific analysis.
- **POST** `/analyze`: Submit text for analysis.
- **PUT** `/analysis/{id}`: Update an analysis.
- **DELETE** `/analysis/{id}`: Delete an analysis.

Uses Axios for asynchronous requests, handling CORS via backend middleware.

---

## Future Enhancements

- Add file upload for text analysis in `AnalysisForm`.
- Implement pagination for `AnalysisList` with large datasets.
- Enhance visualizations with animations or additional chart types (e.g., pie charts for emotions).
