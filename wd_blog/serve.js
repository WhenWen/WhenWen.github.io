import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 7292;
const app = express();

app.use(express.static(path.join(__dirname, 'public')));

// Serve assets folder
app.use('/assets', express.static(path.join(__dirname, 'assets')));

app.listen(PORT, () => {
    console.log(`wd_blog static server listening on http://localhost:${PORT}`);
});
