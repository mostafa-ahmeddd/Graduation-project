import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import logging
from PIL import Image

class ExplainableAI:
    """Your exact Grad-CAM implementation packaged for Flask"""
    
    def __init__(self, model, last_conv_layer_name, img_size=(168, 168)):
        model_path = r"C:/Users/mosta/Downloads/model (1).keras"
        model = keras.models.load_model(model_path)
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.img_size = img_size
        self.logger = logging.getLogger(__name__)

    # Your exact Kaggle-working functions as methods ----------------------------
    
    def get_img_array(self, img_path):
        """Your working preprocessing (grayscale)"""
        img = keras.utils.load_img(img_path, 
                                color_mode='grayscale', 
                                target_size=self.img_size)
        array = keras.utils.img_to_array(img)  # Shape: (h, w, 1)
        array = array / 255.0
        return np.expand_dims(array, axis=0)  # Shape: (1, h, w, 1)

    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """Your exact heatmap generation"""
        grad_model = keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.last_conv_layer_name).output, 
                    self.model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        return tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    def superimpose_heatmap(self, img_path, heatmap, alpha=0.4):
        """Your exact visualization with JET colormap"""
        img = keras.utils.load_img(img_path)
        img = keras.utils.img_to_array(img)
        
        heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap * alpha + img
        return np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Flask integration wrapper ------------------------------------------------
    
    def generate_explanation(self, img_path, class_id=None, output_path=None):
        """
        Your pipeline adapted for Flask
        Args:
            class_id: If None, uses predicted class
            output_path: If provided, saves result image
        Returns:
            Dictionary with results if output_path=None
        """
        try:
            # 1. Preprocess and predict
            img_array = self.get_img_array(img_path)
            preds = self.model.predict(img_array)
            class_id = class_id if class_id is not None else np.argmax(preds[0])
            
            # 2. Generate heatmap
            heatmap = self.make_gradcam_heatmap(img_array, class_id)
            
            # 3. Create visualization (your exact method)
            superimposed = self.superimpose_heatmap(img_path, heatmap)
            
            # Output handling
            if output_path:
                Image.fromarray(superimposed).save(output_path)
            else:
                return {
                    'original': keras.utils.img_to_array(keras.utils.load_img(img_path)),
                    'heatmap': heatmap.numpy(),
                    'superimposed': superimposed,
                    'prediction': {
                        'class_id': class_id,
                        'confidence': float(np.max(preds))
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Explanation failed: {str(e)}")
            raise
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    import sys
    import io

    # Fix Windows console encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    try:
        # 1. Load model
        model_path = r"C:/Users/mosta/Downloads/model (1).keras"
        model = keras.models.load_model(model_path)
        explainer = ExplainableAI(model, last_conv_layer_name="block5_conv4")
        
        # 2. Test with raw string path
        img_path = r"static\uploads\20250622_061001_P_100_HF_.jpg"
        
        # 3. Run explanation
        print("\n=== Running Full Explanation ===")
        explanation_result = explainer.generate_explanation(
            img_path=img_path,
            output_path=r"static\results\explanation.jpg"
        )
        
        # 4. Display results
        print("\n=== Displaying Results ===")
        plt.figure(figsize=(15, 5))
        
        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(Image.open(img_path), cmap='gray')
        plt.title("Original MRI")
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 3, 2)
        img_array = explainer.get_img_array(img_path)
        heatmap = explainer.make_gradcam_heatmap(img_array)
        plt.imshow(heatmap, cmap='viridis')
        plt.title("Raw Heatmap")
        plt.colorbar()
        plt.axis('off')
        
        # Superimposed Explanation
        plt.subplot(1, 3, 3)
        explanation_img = Image.open(r"static\results\explanation.jpg")
        plt.imshow(explanation_img)
        plt.title("Grad-CAM Explanation")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error during testing: {str(e)}", file=sys.stderr)