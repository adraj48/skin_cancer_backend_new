import React, { useState } from "react";
import {
  Box,
  Button,
  Typography,
  LinearProgress,
  Paper,
  Alert,
  Stack,
} from "@mui/material";

// Set VITE_API_URL in Vercel to the Render backend URL. Falls back to the
// local Flask port so `npm run dev` works without any env setup.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(null);
    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    } else {
      setPreview(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setResult({ error: "Please select an image file." });
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.error || `Backend returned ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: `Error connecting to backend: ${error.message}` });
    }
    setLoading(false);
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        backgroundColor: "#f5f5f5",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Paper elevation={3} sx={{ p: 4, width: 400 }}>
        <Typography variant="h4" align="center" gutterBottom>
          Skin Cancer Classifier
        </Typography>
        <Typography variant="body1" align="center" gutterBottom>
          Upload a skin lesion image to predict if it's benign or malignant.
        </Typography>
        <form onSubmit={handleSubmit}>
          <Stack spacing={2}>
            <Button variant="contained" component="label">
              {file ? "Change Image" : "Select Image"}
              <input
                type="file"
                accept="image/*"
                hidden
                onChange={handleFileChange}
              />
            </Button>
            <Button
              type="submit"
              variant="contained"
              color="primary"
              disabled={loading}
            >
              {loading ? "Predicting..." : "Predict"}
            </Button>
            {loading && <LinearProgress />}
          </Stack>
        </form>
        {preview && (
          <Box mt={2} textAlign="center">
            <Typography variant="subtitle1">Preview</Typography>
            <img
              src={preview}
              alt="Preview"
              style={{ maxWidth: "100%", maxHeight: 200 }}
            />
          </Box>
        )}
        {result && (
          <Box mt={2}>
            {result.error ? (
              <Alert severity="error">{result.error}</Alert>
            ) : (
              <Alert severity="success">
                <Typography>
                  <strong>Result:</strong> {result.result.toUpperCase()}
                </Typography>
                <Typography>
                  <strong>Probability:</strong> {Number(result.probability).toFixed(4)}
                </Typography>
              </Alert>
            )}
          </Box>
        )}
      </Paper>
    </Box>
  );
}

export default App;
