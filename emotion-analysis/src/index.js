import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import styled from "styled-components";

const ErrorBoundaryFallback = styled.div`
  color: red;
  padding: 20px;
  background-color: #ffebee;
  border: 1px solid #ffcdd2;
  border-radius: 5px;
  text-align: center;
`;

class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <ErrorBoundaryFallback>
          <h2>Something went wrong!</h2>
          <p>{this.state.error.message}</p>
          <p>Check the console for more details.</p>
          <button onClick={() => window.location.reload()}>Reload Page</button>
        </ErrorBoundaryFallback>
      );
    }
    return this.props.children;
  }
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
