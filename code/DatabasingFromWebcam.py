# OpenCV and Facial Recognition Test App Mk3              16APR2020
# Copyright Doug Hardy and John Granholm

# Creates a folder called Data next to where the Python script is run
# Starts the default webcam
# Uses previous webcam frame to determine if current frame is reliable data
# Captures cleaner data from webcam
# Saves data in machine-friendly bits and human-friendly .jpg files


import face_recognition
import cv2  # required for webcam capture
import os  # listdir lists files found in folder
import numpy as np  # array library
from datetime import datetime, timedelta  # code execution timing
from shutil import copy2, move  # file moving


def AppendDatabase(liveArray, databaseArray, databaseStructure, frameCountTrigger):
    # Checks ProcessFrame's work
    # A face that couldn't be ID by ProcessFrame it comes in as 'ForeignKey' = 0 in liveArray

    # If the face couldn't be ID'd and has been in frame for at least frameCountTrigger frames:
    # AppendDatabase creates a new database record for the new face and assigns
    #   'Key' = database length + 1,
    #   'Name' = 'unknown' + 'Key'
    #   'FaceEncoding' = liveArray current row's 'FaceEncoding'
    # current row in liveArray is updated with the new database row's 'Key' and 'Name' values
    # workingArray is returned as a taller databaseArray

    # Inputs:   liveArray (may contain 'ForeignKey' = 0, 'Name' = '' rows),
    #           databaseArray (a fixed dimm numpy array),
    #           databaseStructure (numpy column names and expected data types - used to keep databaseArray organized)
    #           frameCountTrigger (how soon after a faces appears does AppendDatabase logic fire?)

    # Process:  for each row in liveArray where 'ForeignKey' = 0 and 'FrameCount' >= frameCountTrigger
    #               build a new row for databaseArray
    #               edit liveArray in-line using data from newDatabaseRow
    #               append newDatabaseRow to a workingArray for each new face

    # Returns:  workingArray (a fixed dimm numpy array slightly taller than databaseArray)

    workingArray = databaseArray

    for row in liveArray:
        # If ProcessFrame was unable to ID the face in row['FaceEncoding']
        #   and face has been in frame for at least frameCountTrigger frames
        if row['ForeignKey'] == 0 and row['FrameCount'] >= frameCountTrigger:

            # Build new row of data in the same shape as databaseArray
            newDatabaseRow = np.array([((len(workingArray) + 1), ('Unknown' + str(
                len(workingArray) + 1)), '', row['FaceEncoding'])], databaseStructure)

            # Update liveArray's row['ForeignKey'] with the ['Key'] in databaseArray that contains the data for this person
            row['ForeignKey'] = newDatabaseRow['Key']

            # Update liveArray's row['Name'] with the unknownX assigned by newDatabaseRow's 'Unknown' + len(databaseArray)+1
            row['Name'] = newDatabaseRow[0]['Name']

            # Append newDatabaseRow to the end of workingArray and reinitialize workingArray (slightly taller now)
            workingArray = np.append(workingArray, newDatabaseRow, axis=0)

            # Announce a new row as been added
            print(newDatabaseRow[0]['Name'] + ' appended to database')

    return workingArray


def BuildArray(databaseStructure):
    # Builds databaseArray
    # Checks for pre-built testDatabase2.npy
    # Checks for new .jpgs in ./
    # workingArray becomes databaseArray

    # Inputs:   databaseStructure (numpy column names and expected data types - used to keep databaseArray organized)

    # Process:  if the database exists, load it
    #           if there are pictures in ./
    #               for each picture in ./
    #                   encode, check for similar encodings in database, add new row to database,
    #                       move picture to /Data/UploadedOriginals/ (gets .jpg out of the way for next program launch)

    # Returns:  workingArray (a fixed dimm numpy array - becomes databaseArray)

    # Get the time and date, format it into something database friendly (fixed char count)
    currentTimeAndDate = datetime.now().strftime("%H:%M:%S-%d%b%Y")

    # If database exists, load it
    if os.path.exists('./Data/Database/testDatabase2.npy'):

        workingArray = np.load('./Data/Database/testDatabase2.npy')
        print('\nDatabase load successful!\n')

    # Otherwise initialize an empty workingArray, but be specific on data structure
    else:
        workingArray = np.array([], databaseStructure)
        print('\nCould not find database, building...\n')

    # Load only the .jpg file names in ./ to the knownFaceFiles list
    # File name is assumed to be the name of the person pictured!
    knownFaceFiles = [f for f in os.listdir(
        './') if f.endswith('.jpg')]
    knownFaceFiles.sort()

    print('\nEncoding pictures found in ./\n')

    # Build workingArray from .jpg's found in ./
    for currentFile in knownFaceFiles:

        # Load current image file into a numpy array
        currentImageArray = face_recognition.load_image_file(
            './' + currentFile)

        # Pass the current image numpy array to .face_encodings
        encodedFacesList = face_recognition.face_encodings(currentImageArray)
        # Returns a list of (128)-dimensional face encodings (one per face found)

        # Does supplied image contain a (recognizable) face? How many?

        # If there are no faces found
        if len(encodedFacesList) == 0:
            print('{0:<22}{1}'.format(currentFile, 'No encodable faces found.'))

        # If there is one face found
        if len(encodedFacesList) == 1:
            # This is the only case where naming a face is possible
            # Remember - the identification is driven by the picture's file name!

            # Compare the current face encoding against the database
            facesFound = face_recognition.compare_faces(
                encodedFacesList[0], workingArray['FaceEncoding'])

            # If the encoded face isn't already in the database
            # Aka: if none of the bools in the facesFound list are True
            if not any(facesFound):
                # The encoded face is the primary key
                # When processing the live feed, the .compare_faces list should NEVER contain 2 True values

                # Build new row of data
                newRowOfData = np.array(
                    [((len(workingArray) + 1), currentFile.replace('.jpg', ''), currentTimeAndDate, encodedFacesList[0])], databaseStructure)

                # Add the new row to the end of workingArray
                workingArray = np.append(workingArray, newRowOfData, axis=0)

                print('{0:<22}{1}'.format(currentFile,
                                          'Encoding Success! Moving file to /Data/UploadedOriginals/'))

                try:
                    # Move current .jpg to /Data/UploadedOriginals
                    move('./' + currentFile,
                         './Data/UploadedOriginals/' + currentFile)

                    # Make a folder named from the .jpg file name
                    os.makedirs('./Data/Screenshots/' +
                                currentFile.replace('.jpg', '/'))

                # Handle any and all of the weird reasons a move or mkdr command might fail
                except:
                    print('ERROR: Unable to move ' + currentFile)

            # If the initial .compare_faces came back with a True value
            else:
                # This could get weird in the wild.
                # At the very least we should print the ID results
                print('{0:<22}{1}'.format(currentFile, 'Already in database as ' +
                                          workingArray[facesFound.index(True)]['Name']))
                # This also might be the cleanest line of code I've ever written

        # If too many faces were found
        if len(encodedFacesList) > 1:

            # TODO: 'Would you like to tag the people in this picture (Y/N)?'
            print('{0:<22}{1}'.format(
                currentFile, 'Too many people in picture. People found: ' + str(len(encodedFacesList))))

    # Return all the data sources compiled into one uniform list for live processing (and saving)
    return workingArray


def ClickID(mouseClick, liveArray):
    # Processes user clicks
    # Returns which box in liveArray was clicked, and if it was a good click

    # Inputs:   mouseClick (x/y coordinates of user click)
    #           liveArray (ForignKey points back to databaseArray and x/y coordinates of faces found)

    # Process:  for each row in liveArray
    #               if a click landed in a box labeled 'Unknown'
    #                   Return which row in liveArray was clicked,
    #                       and True for a 'user input that triggers response' signal
    #               if the click landed anywhere else
    #                   Return 0, False

    for rowIndex, row in enumerate(liveArray):

        # print(row['FaceLocation'])
        # returns [ 76 225 166 135]
        # top, right, bottom, left

        # print(row['Name'][0:7])
        # returns Unknown

        xCord = mouseClick[0]
        yCord = mouseClick[1]

        top = row['FaceLocation'][0] * 2
        right = row['FaceLocation'][1] * 2
        bottom = row['FaceLocation'][2] * 2
        left = row['FaceLocation'][3] * 2

        # print(row['Name'][0:7])

        # If the mouseclick landed within one of the found faces in liveArray
        if xCord > left and xCord < right and yCord > top and yCord < bottom:

            # And the mouse click landed in a non-ID'd box
            if row['Name'][0:7] == 'Unknown':

                return rowIndex, True

    return 0, False


def PaintBoxes(inputFrame, liveArray):
    # Paints names and boxes on inputFrame
    # Uses liveArray ['FaceLocation'] x/y cords to paint boxes on a frame
    # Uses liveArray ['Name'] to paint an ID'd name below box

    # Inputs:   inputFrame (numpy array representing pixels)
    #           liveArray (x/y coordinates and name data)

    # Process:  draw box at ['FaceLocation'], draw ['Name'] below it

    # Returns:  void

    for row in liveArray:

        # Scale back up face locations - processing was scaled to 1/2 size
        top = row['FaceLocation'][0] * 2
        right = row['FaceLocation'][1] * 2
        bottom = row['FaceLocation'][2] * 2
        left = row['FaceLocation'][3] * 2

        # Draw a box around the face
        cv2.rectangle(inputFrame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(inputFrame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(inputFrame, row['Name'], (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)


def ProcessFrame(inputFrame, lastFrameArray, databaseArray, liveDataStructure, databaseRecheckTrigger):
    # Builds workingArray from inputFrame
    # ID's faces in workingArray using multiple sources (lastFrameArray, databaseArray),
    #   organized by processor cost
    # Assign a name to a face encoding found in frame
    # Any faces that can't be ID'd by the last frame's data or the database
    #   remain in workingArray as initialized: 'ForeignKey' = 0, 'Name' = 'Unknown'

    # Inputs:   inputFrame (the cv2 webcam capture)
    #           lastFrameArray (a copy of last frame's liveArray aka - the work ProcessFrame did last frame)
    #           databaseArray (a fixed dimm numpy array),
    #           liveDataStructure (numpy column names and expected data types - used to keep liveArray organized)

    # Process:  for each face found in inputFrame
    #               build a new workingArray row
    #                   set 'ForeignKey' = 0, 'Name' = 'Unknown', capture face location, and face encoding data
    #           for each in workingArray
    #               check lastFrameArray for matching faces
    #                   set 'ForeignKey' and 'Name' if a match comes back True
    #               if lastFrameArray cannot ID the face (or if data is older than databaseRecheckTrigger)
    #                   check databaseArray for matching faces
    #                       set 'ForeignKey' and 'Name' if a match comes back True
    #                       (some rows might remain 'ForeignKey' = 0, 'Name' = 'Unknown')

    # Returns:  workingArray (a fixed dimm numpy array - becomes liveArray)

    def CheckLastFrame(row, inputEncoding):
        # Checks inputEncoding against lastFrameArray's 'FaceEncoding' data

        # if a match is found and last frame data is less than databaseRecheckTrigger frames old
        #   set workingArray[row] data to lastFrameArray's matched index
        #   return True

        lastFrameMatches = face_recognition.compare_faces(
            lastFrameArray['FaceEncoding'], inputEncoding)

        # If this frame contains the same face as last frame
        if True in lastFrameMatches:

            # Get the index of the true value
            indexOfTrue = lastFrameMatches.index(True)

            # If 'FrameCount' data in lastFrameArray is greater than databaseRecheckTrigger
            # Aka if this row in lastFrameArray has been used to ID the current frame
            #   too many times, force a database check
            if lastFrameArray[indexOfTrue]['FrameCount'] >= databaseRecheckTrigger:

                # Print debug line and return CheckLastFrame as false
                # Forces CheckDatabase function to fire
                # print('database recheck')
                return False

            # Get the name from the hard work we did last frame
            workingArray[row]['ForeignKey'] = lastFrameArray[indexOfTrue]['ForeignKey']
            workingArray[row]['Name'] = lastFrameArray[indexOfTrue]['Name']

            # Iterate 'FrameCount' if this face is ID'd from last frame's data
            workingArray[row]['FrameCount'] = lastFrameArray[indexOfTrue]['FrameCount'] + 1

            # print(workingArray[row]['FrameCount'])

            return True

    def CheckDatabase(row, inputEncoding):
        # Checks inputEncoding against databaseArray's 'FaceEncoding' data
        # If a match can't be found here, ProcessFrame will return this row of workingArray
        #   as 'ForeignKey' = 0, 'Name' = 'Unknown' (unedited after initialization)

        # if a match is found
        #   set workingArray[row] data to databaseArray's matched index

        databaseMatches = face_recognition.compare_faces(
            databaseArray['FaceEncoding'], currentFaceEncoding)

        # If a match was found, add it to the thisFrameNames list
        if True in databaseMatches:

            indexOfTrue = databaseMatches.index(True)

            workingArray[row]['ForeignKey'] = databaseArray[indexOfTrue]['Key']
            workingArray[row]['Name'] = databaseArray[indexOfTrue]['Name']

    workingArray = np.array([], liveDataStructure)

    # Resize frame to 1/2 size for faster face recognition processing
    smallFrame = cv2.resize(inputFrame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgbSmallFrame = smallFrame[:, :, ::-1]

    # Check rgbSmallFrame for faces (high cost function!)
    faceLocations = face_recognition.face_locations(
        rgbSmallFrame)

    # If faces are found
    if len(faceLocations) > 0:

        # Get face encodings at each faceLocations' x/y cord
        faceEncodings = face_recognition.face_encodings(
            rgbSmallFrame, faceLocations)

        # Combine faceLocations, faceEncodings for easier iterating
        faceData = zip(faceLocations, faceEncodings)

        # For each faceLocation, faceEncoding in faceData
        for faceLoc, faceEnc in faceData:

            # Create a new workingArray row using the liveDataStructure column names and data types
            newLiveRow = np.array(
                [(0, 'Unknown', 1, faceLoc, faceEnc)], liveDataStructure)

            # Append newLiveRow to workingArray and reinitialize workingArray (slightly taller now)
            workingArray = np.append(workingArray, newLiveRow, axis=0)

    # For each item in workingArray's 'FaceEncoding' column
    for currentRow, currentFaceEncoding in enumerate(workingArray['FaceEncoding']):

        # If there was a face last frame, check the last frame data first
        if len(lastFrameArray) > 0:

            # Check last frame, returns True if it finds a match
            if not CheckLastFrame(currentRow, currentFaceEncoding):

                # Check databaseArray for matches
                CheckDatabase(currentRow, currentFaceEncoding)
                # Iterating across a large database is expensive!
                # This code only fires when CheckLastFrame returns false

        # If there was NO face last frame
        else:

            # Check databaseArray for matches
            CheckDatabase(currentRow, currentFaceEncoding)

    # Return processing results
    return workingArray


def PromoteUnknown(newNameInput, liveArrayRow, liveArray, databaseArray):
    # Handles folder renaming
    # Updates liveArray
    # Updates databaseArray

    # Inputs:   newNameInput (user's text input)
    #           liveArrayRow (which box in liveArray the user clicked on - found by ClickID)
    #           liveArray (for retrieving databaseArray's key and editing liveArray name data)
    #           databaseArray (editing databaseArray name data)

    # Process:  if user input is not blank
    #               use liveArray's 'ForeignKey' to point to a row in databaseArray

    #               use the 'Name in databaseArray to set the current file path
    #               use the user's text input to set the new file path
    #               rename file path

    #               update 'Name in liveArray with user's text input
    #               update 'Name in databaseArray with user's text input

    # Returns:  void

    if newNameInput != '' and newNameInput.isalpha():

        databaseRow = liveArray[liveArrayRow]['ForeignKey'] - 1

        currentName = databaseArray[databaseRow]['Name']

        currentFilePath = './Data/Screenshots/' + \
            databaseArray[databaseRow]['Name']

        newFilePath = './Data/Screenshots/' + newNameInput

        try:
            os.rename(currentFilePath, newFilePath)

            liveArray[liveArrayRow]['Name'] = newNameInput

            databaseArray[databaseRow]['Name'] = newNameInput

            print(currentName +
                  ' updated in database and /Data/Screenshots/ to ' + newNameInput)

        except:

            print('Unable to update record ' + currentName)

    else:
        print('Unacceptable characters used - unable to update record')


def SaveArray(databaseArray):
    # Saves databaseArray
    # Prints a report of what's being saved

    # Inputs:   databaseArray (a fixed dimm numpy array)

    # Process:  for each row in databaseArray
    #               print 'Key', 'Name', 'FrameSaved', type('FaceEncoding')
    #           save databaseArray as testDatabase2.npy

    # Returns:  void

    # Prove there's data in the array before saving
    print('\n\nDatabase contents just before saving:\n')
    print('{0:3}  {1:<18} {2:<22} {3}\n'.format(
        'Key', 'Name', 'FrameSaved', "type('FaceEncoding')"))

    # Print each row of data in databaseArray
    for currentRow in databaseArray:
        print('{0:3}  {1:<18} {2:<22} {3}'.format(
            currentRow['Key'], currentRow['Name'], currentRow['FrameSaved'], str(type(currentRow['FaceEncoding']))))
        # Take the one non-string item 'FaceEncoding' and represent it as an object type
        # Aka: it's there - I promise!

    # Minimalist sanity check
    print('\nArray length: ' + str(len(databaseArray)))

    # Save array as a binary file (maintains float values)
    np.save('./Data/Database/testDatabase2.npy', databaseArray)


def TakeScreenshots(inputFrame, liveArray, databaseArray, screenShotInterval, frameCountTrigger):
    # Checks databaseArray's 'FrameSaved' at liveArray's 'ForeignKey'
    # 'FrameSaved' is converted to datetime object
    # If 'FrameSaved' is older than now by more than 1 minute
    #   and face has been in frame for at least frameCountTrigger frames:
    # Save timestamped .jpg to /Data/Screenshots/ liveArray's 'Name' /
    # TODO: crop screenshot down to liveArray's 'FaceLocation'
    #       OR paint one box?
    #           would involve reworking PaintBoxes into PaintBox and TakeScreenshots into TakeScreenshot
    #               and iterating across liveArray in main

    # Inputs:   inputFrame (numpy array representing pixels)
    #           liveArray (contains currently found face data)
    #           databaseArray (contains data on when last screenshot was taken)
    #           frameCountTrigger (how soon after a faces appears does TakeScreenshots logic fire?)

    # Process:  for each row in liveArray
    #               check databaseArray's 'FrameSaved' at liveArrayrow's 'ForeignKey'
    #               take screenshot if last screenshot is too old
    #               update databaseArray with new 'FrameSaved' data

    # Returns:  void

    # Get the time and date, format it into something database friendly (fixed char count)
    currentTimeAndDate = datetime.now().strftime("%H:%M:%S-%d%b%Y")

    def SaveJPG(inputName):

        filePath = './Data/Screenshots/' + inputName + '/'

        try:

            # If the named folder doesn't exist in /Screenshots
            if not os.path.exists(filePath):
                # Make named folder
                os.makedirs(filePath)

            # Save date and time stamped .jpg
            cv2.imwrite(filePath + currentTimeAndDate +
                        '.jpg', inputFrame)

            # Update databaseArray's 'FrameSaved' to process against next time face appears in frame
            databaseArray[databaseRow]['FrameSaved'] = currentTimeAndDate

            print('Screenshot of ' + inputName +
                  ' saved and FrameSaved timestamp updated.')

        # Handle any and all of the weird reasons a mkdr or save .jpg command might fail
        except:
            print('ERROR: Unable to save screenshot for ' + inputName)

    for row in liveArray:

        # If face has been in frame for at least frameCountTrigger frames
        if row['FrameCount'] >= frameCountTrigger:

            # liveArray contains a reference (forign key) to the databaseArray
            # The '-1': the key for the first row is 1, but the INDEX of the first row is 0
            databaseRow = int(row['ForeignKey']-1)

            # If a screenshot has never been saved, take Screenshot (new database record)
            if databaseArray[databaseRow]['FrameSaved'] == '':
                SaveJPG(databaseArray[databaseRow]['Name'])

            # If a screenshot has been saved before, get the date it was save from databaseArray's 'FrameSaved'
            else:
                lastSave = datetime.strptime(
                    databaseArray[databaseRow]['FrameSaved'], "%H:%M:%S-%d%b%Y")

                # If the most recent screenshot is older than X minutes
                if datetime.now() >= lastSave + timedelta(seconds=screenShotInterval):

                    SaveJPG(databaseArray[databaseRow]['Name'])


def ClickedInWindow(event, x, y, flags, param):
    # This function catches everything cv2.setMouseCallback spits out.
    # Of all the data .setMouseCallback returns,
    #   this ClickedInWindow function is only interested in one event: a left mouse click

    # Reference a variable from outside this function (public), don't create a new variable (private)
    global mouseClick
    # global variables are usually bad, but might be necessary
    #   because of how this tool is used to capture cv2.setMouseCallback data
    #   (no opportunity to set an external variable)

    if event == cv2.EVENT_LBUTTONDOWN:

        # If a left mouse button click is detected, set the global mouseClick
        #   to the current click's x/y coordinates.
        mouseClick = [x, y]


# Setup timing
startTime = datetime.now()

# Define the 'column' names and data type of the database and live arrays
liveDataStructure = np.dtype(
    [('ForeignKey', 'uint32'), ('Name', 'U15'), ('FrameCount', 'uint32'), ('FaceLocation', 'uint32', (4)), ('FaceEncoding', 'float64', (128))])

databaseStructure = np.dtype(
    [('Key', 'uint32'), ('Name', 'U15'), ('FrameSaved', 'U18'), ('FaceEncoding', 'float64', (128))])


# Initialize global variables
lastFrameArray = np.array([], liveDataStructure)
processThisFrame = True
userClickedOnUnknown = False
mouseClick = [-1, -1]
screenShotInterval = 60     # Measured in seconds
frameCountTrigger = 2       # How soon after a faces appears
#                               does AppendDatabase and TakeScreenshots logic fire?
databaseRecheckTrigger = 5  # How many frames can ProcessFrame use lastFrameArray's
#                               data before a database recheck?
#                             Prevents liveArray from latching onto a bad ID (for too long)

# Check for and create file folders
if not os.path.exists('./Data/'):
    os.makedirs('./Data/')
    os.makedirs('./Data/Database/')
    os.makedirs('./Data/UploadedOriginals/')
    os.makedirs('./Data/Screenshots/')

# Terminal output
print('\nOpenCV and Facial Recognition Test App Mk3\n')
print('This program looks for pictures of people in ./')
print("The name of the file is assumed to be the pictured person's name.")
print('This program can also be updated with new faces once they appear in the webcam.')

# Load .npy file and any new .jpg's to RAM
databaseArray = BuildArray(databaseStructure)

# Get a reference to the default webcam
videoCapture = cv2.VideoCapture(0)

print('\nSetup time: ' + str(datetime.now() - startTime))

# Open new Qt window.
# This is normally done with .imshow('Video', frame)
#   but for .setMouseCallback to work it needs a named window.
cv2.namedWindow('Video')

# Listen for mouse events, if any happen push to ClickedInWindow
cv2.setMouseCallback('Video', ClickedInWindow)

# More terminal output
print('\n\n...\n')
print('\nLaunching OpenCV window.')
print('\nProgram instructions:')
print('    1. Click on an Unknown face to tag that person.\n')
print('    2. Press q to quit!\n\n')

# The 'main' or 'live' function
while True:

    # FPS timing
    newTime = datetime.now()

    # Read a frame of video
    ret, frame = videoCapture.read()

    # Process the faces in the frame and return an array row for each face found in frame
    liveArray = ProcessFrame(
        frame, lastFrameArray, databaseArray, liveDataStructure, databaseRecheckTrigger)

    # For each row in liveArray that leaves ProcessFrame without a successful ID, create a new database row
    databaseArray = AppendDatabase(
        liveArray, databaseArray, databaseStructure, frameCountTrigger)

    # Add .jpgs to image database on timed intervals, per face
    TakeScreenshots(frame, liveArray, databaseArray,
                    screenShotInterval, frameCountTrigger)

    # Use the x/y cords and name of the found face to display the results on frame (building GUI)
    PaintBoxes(frame, liveArray)

    # Paint FPS data on frame
    difference = (datetime.now() - newTime)
    fps = 'FPS: {:0.2f}'.format(1 / difference.microseconds * 1000000)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, fps, (0, 30), font, 1, (0, 0, 255), 2)

    # Save ProcessFrame's work from this frame to help it ID against a smaller list next frame
    lastFrameArray = liveArray

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Listen for user click, if click happens in an UnkwownX box, return which box was clicked on and a True flag
    liveArrayRow, userClickedOnUnknown = ClickID(mouseClick, liveArray)

    # If ClickID returned True
    if userClickedOnUnknown == True:

        # This line pauses the while loop until user inputs text
        newNameInput = input('Tag this person: ')

        # Updates record in liveArray, databaseArray, and record's /Screenshot/ folder
        PromoteUnknown(newNameInput, liveArrayRow, liveArray, databaseArray)

        # Reset user click to impossible coordinates
        mouseClick = [-1, -1]

        # Reset while loop to run indefinitely
        userClickedOnUnknown = False

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
videoCapture.release()
cv2.destroyAllWindows()

# Print and save the databaseArray in its final state before program exit
SaveArray(databaseArray)
