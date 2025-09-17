import express from 'express';
import path from 'path';
const app = express();
import { fileURLToPath } from 'url';
const PORT = 8080;

import dotenv from 'dotenv';
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Serve static files
app.use('/static', express.static(path.join(__dirname, 'static')));

// Serve HTML
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/models/:filename', (req, res) => {
    const filename = req.params.filename;
    const filePath = path.join(__dirname, 'models', filename);

    // Set the correct MIME type for VRM (binary)
    res.sendFile(filePath, { headers: { 'Content-Type': 'application/octet-stream' } }, (err) => {
        if (err) {
            console.error('Error sending file:', err);
            res.status(404).send('File not found');
        }
    });
});

app.post('/api/text2motion', async (req, res) => {
  try {
    const body = req.body; // { prompt, target_skeleton }

    const response = await fetch('https://api.text2motion.ai/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-apikey': process.env.text_to_motion__api_key
      },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      return res.status(response.status).json({ error: response.statusText });
    }

    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error' });
  }
});

app.use(express.json());

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
