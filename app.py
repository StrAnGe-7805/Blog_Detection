from flask import Flask, request, jsonify
from model import predict

app = Flask(__name__)

@app.route('/api/test', methods=['POST'])
def test():

    image_file_names = []

    images = request.files.to_dict() #convert multidict to dict
    for image in images:     #image will be the key 
        file_name = images[image].filename
        image_file_names.append(file_name)
        images[image].save('components/recieved_images/'+file_name)
    text = request.form.get('text')
    result = predict(image_file_names,text)
    response = {'result': result}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)