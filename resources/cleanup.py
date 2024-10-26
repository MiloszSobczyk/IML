import os

def removeHiddenFiles(rootDir):
    for dirPath, _, filenames in os.walk(rootDir):
        for fileName in filenames:
            if fileName.startswith('.') and 'git' not in fileName:
                filePath = os.path.join(dirPath, fileName)
                os.remove(filePath)
                print(filePath)

removeHiddenFiles('./')