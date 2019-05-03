from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads,IMAGES
from flask import Flask, request, redirect, url_for, send_from_directory,render_template
from werkzeug import secure_filename
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import shutil
import os
import subprocess
from shutil import copyfile

UPLOAD_FOLDER = '/Users/prasadgujar16/Desktop/searching/my-images'
ALLOWED_EXTENSIONS = set([ 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home():
	return render_template('indexs.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = "a.jpg"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('indexs.html')
    return  '''
        <!doctype html>
        <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <meta name="description" content="">
    <meta name="author" content="">
    <link rel="icon" href="favicon.ico">

    <title>Upload a New File</title>

    <!-- Bootstrap core CSS -->
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" rel="stylesheet">

    <!-- Custom styles for this template 
    <link rel="stylesheet" type="text/css" href="templates/jumbotron.css"> -->
     <style>
       /* Move down content because we have a fixed navbar that is 3.5rem tall */
body {
  padding-top: 3.5rem;
  background-image: url("/static/blog.jpg")
  
}
.jumbotron{
  background-image: url("/static/blog.jpg")
  background-background-repeat: none;
}
</style>
  </head>>
  <body>

    <nav class="navbar navbar-expand-md navbar-dark fixed-top bg-dark">
      <a class="navbar-brand" href="#">CapSearch</a>
      
          </li>
        </ul>
        <form class="form-inline my-2 my-lg-0" action="result" method="POST">
          <input class="form-control mr-sm-2" type="text" placeholder="Search" aria-label="Search" name="Name">
          <input type="submit" value="submit">
        </form>
      </div>
    </nav>

    <main role="main">

      <!-- Main jumbotron for a primary marketing message or call to action -->
      <div class="jumbotron">
        <div class="container">
          <h1 class="display-3">CapSearch</h1>
          
        </div>
      </div>

       <form action="" method=post enctype=multipart/form-data>
        <p><input type=file name=file>
        <input type=submit value=Upload>
        </form>

          
       
        <hr>

      </div> <!-- /container -->

    </main>
       
        '''

@app.route('/show/<filename>')
def uploaded_file(filename):
    filename = 'http://127.0.0.1:5000/uploads/' + filename
    return render_template('index.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/rip',methods=['POST','GET'])
def ok():
    hists = os.listdir('static/people_photo/')
    hists = ['people_photo/' + file for file in hists]
    return render_template('index.html', hists = hists)

photos = UploadSet('photos', IMAGES)

##

app.config['UPLOADED_PHOTOS_DEST'] = 'ok/img'
configure_uploads(app, photos)

'''
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        return render_template('indexs.html')
    return render_template('upload.html')'''


@app.route('/caption')
def content():
	
	inputs = "/Users/prasadgujar16/Desktop/searching/my-images/a.jpg"
	'''output = "/Users/prasadgujar16/Desktop/searching/my-images/"
	if os.path.exists(output):
		os.remove(output)
	try:
		os.rename(inputs,output)
	except OSError:
		pass
	#os.system("main_cap.py")'''
	subprocess.call(" python extest.py 1", shell=True)
	text = open('caption.txt', 'r+')
	content = text.read()
	text.close()
	rename = ""
	with open('caption.txt','r') as data:
		for line in data:
			rename +=line
			rename +=" "
	data.close()
	rename = rename[:-1]
	k1 = '/Users/prasadgujar16/Desktop/searching/my-images/'
	cname = k1 + rename + ".jpg"
	os.rename(inputs,cname)
	output = "/Users/prasadgujar16/Desktop/searching/ok/img"
	shutil.copy(cname,output)
	return render_template('content.html', text=content)

@app.route('/result',methods=['POST','GET'])
def result():
	name = request.form.get('Name')
	print (name)
	print ("nahi aya")
	cur = str(name)
	if request.method=='POST' and name:
		with open('query.txt', 'w') as f:
			print ("aya")
			f.write(cur)
			f.close()
			subprocess.call(" python search.py 1", shell=True)
		return render_template("indexs.html")
	return render_template("indexs.html")
if __name__ == '__main__':
	app.run(debug=True)