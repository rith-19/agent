from flask import Flask, render_template, request, jsonify
import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

class ReactAppGenerator:
    def __init__(self):
        # Initialize Groq LLM with proper configuration
        self.llm = ChatGroq(
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="groq/llama-3.1-70b-versatile"
        )
        
        # Initialize single agent for both generation and refinement
        self.agent = Agent(
            role="Senior React Application Developer",
            goal="Generate and optimize production-ready React applications",
            backstory="""You are a highly experienced React developer specializing in building 
                     production-grade applications. Your expertise covers modern React practices, 
                     component architecture, state management, optimization, and performance tuning. 
                     You excel at creating well-structured, maintainable, and efficient React applications 
                     while following industry best practices.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def create_app_task(self, app_name, description):
        return Task(
            description=f"""Create a complete, optimized React application following these requirements:
                
                Application Name: {app_name}
                Requirements: {description}
                
                Provide a production-ready application including:
                1. Full component file structure with optimal organization and code
                2. Complete, optimized code for each component
                3. Efficient state management implementation
                4. Clean, maintainable styling (CSS/SCSS)
                5. Comprehensive package.json with all required dependencies
                6. Detailed README.md with setup and usage instructions
                7. Proper error handling and edge case management
                8. Performance optimizations and best practices
                9. Clear documentation and code comments
                
                The code should follow React best practices, implement proper error boundaries, 
                use modern React patterns, and be ready for production deployment.""",
            agent=self.agent,
            expected_output="A complete, optimized React application codebase including all components, styling, configuration files, and documentation."
        )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_app():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        app_name = data.get('app_name')
        description = data.get('description')

        if not app_name or not description:
            return jsonify({'error': 'Missing required fields'}), 400

        if not os.getenv("GROQ_API_KEY"):
            return jsonify({'error': 'GROQ_API_KEY not found in environment variables'}), 500

        # Initialize generator and create task
        generator = ReactAppGenerator()
        task = generator.create_app_task(app_name, description)

        # Create and run crew with single task
        crew = Crew(
            agents=[generator.agent],
            tasks=[task]
        )

        # Generate and optimize application in one go
        result = crew.kickoff()

        return jsonify({
            'status': 'success',
            'code': str(result)
        })

    except Exception as e:
        return jsonify({'error': f'Error during generation: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)


