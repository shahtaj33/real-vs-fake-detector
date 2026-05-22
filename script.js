let detectorModel;
const statusEl = document.getElementById('status');
const uploaderEl = document.getElementById('imageUploader');
const previewEl = document.getElementById('preview');
const resultEl = document.getElementById('result');

async function loadBrowserEngine() {
    try {
        // Point the compiler to the CDN location where WebAssembly binaries live
        tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/');
        
        // Fetch your local model file relative to Vercel's root directory output
        detectorModel = await tflite.loadTFLiteModel('./model.tflite');
        
        statusEl.innerText = "⚡ Model Loaded! Device hardware is ready.";
        statusEl.className = "ready";
        uploaderEl.disabled = false;
    } catch (err) {
        statusEl.innerText = "Initialization failed: " + err.message;
        console.error(err);
    }
}

uploaderEl.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Show image preview layout
    previewEl.src = URL.createObjectURL(file);
    previewEl.style.display = 'block';
    resultEl.innerText = "Analyzing pixel data...";

    previewEl.onload = async () => {
        try {
            // 1. Convert HTML image element directly into a raw Pixel Tensor
            const rawTensor = tf.browser.fromPixels(previewEl);

            // 2. Preprocess: Resize image to 224x224 (or whatever your model requires) and normalize
            const processedTensor = tf.tidy(() => {
                return tf.image.resizeBilinear(rawTensor, [224, 224]) // Match your model shape!
                    .toFloat()
                    .div(255.0)  // Scaling pixels from 0-255 down to 0.0-1.0 range
                    .expandDims(0); // Add batch dimension [1, 224, 224, 3]
            });

            // 3. Run client-side prediction entirely on the user's device
            const outputTensor = detectorModel.predict(processedTensor);
            const predictionData = await outputTensor.data();

            // 4. Output array rendering
            resultEl.innerText = `Raw Model Score output: ${Array.from(predictionData).map(n => n.toFixed(4)).join(', ')}`;
            
            // Clean up memory leaks 
            rawTensor.dispose();
            processedTensor.dispose();
            outputTensor.dispose();

        } catch (error) {
            resultEl.innerText = "Inference Error: " + error.message;
        }
    };
});

// Boot application
loadBrowserEngine();
