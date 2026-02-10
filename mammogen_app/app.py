import os
import base64
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Note: For this Client-based approach, we don't use genai.configure()
# The API key is passed directly when creating the client.

def build_prompt(data):
    """Dynamically constructs a detailed prompt from form data."""
    
    prompt_parts = [
        f"A photorealistic, high-resolution {data.get('imaging_tech', '2D digital mammogram')} "
        f"of the {data.get('breast_side', 'left')} breast in a {data.get('mammogram_view', 'Mediolateral Oblique (MLO)')} view."
    ]

    # Patient Details
    if data.get('age'):
        prompt_parts.append(
            f"The patient is a {data.get('age')}-year-old {data.get('menopausal_status', 'post-menopausal')} woman."
        )

    # Tissue Attributes
    prompt_parts.append(
        f"The breast tissue demonstrates {data.get('breast_density', 'BI-RADS C: Heterogeneously dense')} density."
    )
    if data.get('pectoral_muscle') == 'yes':
        prompt_parts.append("The pectoral muscle is well-visualized.")
    if data.get('nipple_profile') == 'yes':
        prompt_parts.append("The nipple is seen in profile.")
    
    # Pathological Findings
    if data.get('include_pathology') == 'on':
        findings = []
        # Mass description
        if data.get('has_mass') == 'on':
            mass_desc = (
                f"a {data.get('mass_size', '1.5 cm')} {data.get('mass_density', 'high-density')}, "
                f"{data.get('mass_shape', 'irregular')} mass with {data.get('mass_margins', 'spiculated')} margins "
                f"is present in the {data.get('mass_location', 'upper outer quadrant')}."
            )
            findings.append(mass_desc)
        
        # Calcification description
        if data.get('has_calcifications') == 'on':
            calc_desc = (
                f"There is a {data.get('calc_distribution', 'grouped')} cluster of "
                f"{data.get('calc_morphology', 'fine pleomorphic')} microcalcifications."
            )
            findings.append(calc_desc)
        
        # Architectural Distortion
        if data.get('has_distortion') == 'on':
            findings.append("Subtle architectural distortion is also noted.")

        if findings:
            prompt_parts.append("Key findings include: " + " ".join(findings))
        else:
            prompt_parts.append("No specific pathological findings were requested, but the scan should appear clinically realistic.")

    else:
        prompt_parts.append("No masses, suspicious calcifications, or architectural distortion are present.")

    # Final styling
    prompt_parts.append(
        f"The image should have {data.get('image_quality', 'optimal exposure and contrast')} and be displayed "
        f"in {data.get('image_display', 'standard inverted grayscale')}."
    )
    prompt_parts.append("This is a medical X-ray style image, not a photograph.")
    
    return " ".join(prompt_parts)


def generate_image_from_gemini(prompt):
    """Connects to the Gemini API using the Client for the preview model."""
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        model = "gemini-2.5-flash-image-preview"
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        )

        text_response_parts = []
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            # Check for and immediately return the image data when found
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    return image_data, mime_type
            
            # If no image, collect text parts for a potential error message
            if chunk.text:
                text_response_parts.append(chunk.text)

        # If the loop finishes without finding an image, return the collected text as an error.
        if text_response_parts:
            return None, "".join(text_response_parts)
        
        return None, "Image generation failed. No image data or text error was returned from the API."

    except Exception as e:
        print(f"An exception occurred: {e}")
        return None, str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    form_data = request.form.to_dict()
    
    prompt = build_prompt(form_data)
    print(f"Generated Prompt: {prompt}")

    image_bytes, result = generate_image_from_gemini(prompt)

    if image_bytes:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = result
        return jsonify({
            'status': 'success',
            'image': base64_image,
            'mime_type': mime_type
        })
    else:
        return jsonify({'status': 'error', 'message': result})


if __name__ == '__main__':
    app.run(debug=True)