from flask import Flask, request, jsonify
from generate import init_db, generate

app = Flask(__name__)

# Initialize DB when server starts
init_db()

@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint.
    Expects JSON: { "user_id": "user123", "query": "What is LangChain?" }
    """
    data = request.json
    user_id = data.get("user_id")
    query = data.get("query")

    if not user_id or not query:
        return jsonify({"error": "user_id and query are required"}), 400

    # Generate answer (fetches chat history internally)
    result = generate(user_id, query)

    return jsonify({
        "user_id": user_id,
        "query": query,
        "source": result["source"],
        "answer": result["answer"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
