import os
import base64
import json
import logging
from openai import OpenAI
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EventDescriptionAgent:
    def __init__(self, examples_path=None):
        # Initialize OpenAI client compatible with DashScope
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen3-omni-flash"
        self.examples = self._load_examples(examples_path) if examples_path else []

    def _load_examples(self, examples_path):
        try:
            with open(examples_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load examples: {e}")
            return []

    def _encode_image(self, image_path):
        """Encodes an image to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _load_data(self, folder_path):
        """Loads text and images from the specified folder."""
        texts = []
        images = []
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        files = sorted(os.listdir(folder_path))
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                logging.info(f"Loading image: {file}")
                base64_image = self._encode_image(file_path)
                images.append(base64_image)
            elif file.lower().endswith('.txt'):
                logging.info(f"Loading text: {file}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        
        return "".join(texts), images

    def analyze_event(self, folder_path):
        """
        Analyzes the event data in the given folder and returns a structured description (List[Dict]).
        """
        logging.info(f"Processing event folder: {folder_path}")
        text_context, images = self._load_data(folder_path)
        
        system_content = "你是一个专业的突发事件应急分析助手。你的任务是根据提供的现场图片和文字描述，生成一条简洁、客观且概括得当的突发事件描述文本。描述应包含关键要素（如时间、地点、事件类型、现场状况等），便于快速通报。"
        
        if self.examples:
            system_content += f"请严格参考以下JSON格式（仅参考格式字段，严禁照抄示例内容）：{json.dumps(self.examples, ensure_ascii=False, indent=2)}请务必根据实际输入的图片和文字进行分析。如果图片中没有相关信息，请不要编造。输出必须是合法的JSON字符串，不要包含markdown代码块标记。"

        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": []
            }
        ]

        # Add images to the message
        for img_b64 in images:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                },
            })

        # Add text context to the message
        if text_context:
            messages[1]["content"].append({
                "type": "text",
                "text": f"以下是关于该突发事件的文字描述信息：{text_context}请结合图片和文字，生成该突发事件的概括性描述。"
            })
        else:
            messages[1]["content"].append({
                "type": "text",
                "text": "请根据提供的图片，生成该突发事件的概括性描述。"
            })

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                modalities=["text"],
                stream=False
            )
            content = completion.choices[0].message.content
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Parse JSON
            try:
                parsed_result = json.loads(content)
                return parsed_result
            except json.JSONDecodeError as je:
                logging.error(f"JSON parsing failed. Raw content: {content}")
                return {"error": "JSONDecodeError", "raw_content": content}
                
        except Exception as e:
            logging.error(f"Error during generation: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Example usage
    base_inputs_dir = "/home/xiaohengli/Downloads/wenjie_AI/multimodel/multi_inputs"
    examples_file = "/home/xiaohengli/Downloads/wenjie_AI/multimodel/examples.json"
    agent = EventDescriptionAgent(examples_path=examples_file)

    if os.path.exists(base_inputs_dir):
        # Iterate through timestamped subfolders
        subfolders = [f.path for f in os.scandir(base_inputs_dir) if f.is_dir()]
        subfolders.sort() # Process in order
        
        if not subfolders:
            print(f"No subfolders found in {base_inputs_dir}")
        
        for folder in subfolders:
            print("-" * 50)
            result = agent.analyze_event(folder)
            # Pretty print the Python object
            print(f"Result for {os.path.basename(folder)}:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Input directory not found: {base_inputs_dir}")
