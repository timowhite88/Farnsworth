"""
Jinja2 template for rendering community highlights.
"""

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Community Highlights</title>
    <style>
        body { font-family: Arial, sans-serif; }
        h1 { color: #333; }
        ul { list-style-type: none; padding: 0; }
        li { margin-bottom: 10px; border-bottom: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>Community Highlights</h1>
    <ul>
        {% for highlight in highlights %}
            <li><strong>{{ highlight.title }}</strong>: {{ highlight.summary }}</li>
        {% endfor %}
    </ul>
</body>
</html>