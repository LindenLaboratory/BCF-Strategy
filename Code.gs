//@OnlyCurrentDoc
function onOpen(e) {
  var ui = SpreadsheetApp.getUi();
  ui.createMenu("Import CSV data")
    .addItem("Import from URL", "importCSVFromUrl")
    .addItem("Import from Drive", "importCSVFromDrive")
    .addToUi();
}

// Displays an alert as a Toast message
function displayToastAlert(message) {
  SpreadsheetApp.getActive().toast(message, "Alert");
}

// Writes data to sheet with provided sheetName (without .csv extension)
function writeDataToSheet(data, sheetName) {
  var ss = SpreadsheetApp.getActive();
  var formattedSheetName = sheetName.replace(/\.csv$/, ""); // Remove .csv extension
  var sheet = ss.insertSheet(formattedSheetName);
  sheet.getRange(1, 1, data.length, data[0].length).setValues(data);
  return sheet.getName();
}

// Prompts user for input
function promptUserForInput(promptText) {
  var ui = SpreadsheetApp.getUi();
  var prompt = ui.prompt(promptText);
  var response = prompt.getResponseText();
  return response;
}

// Imports CSV data from URL
function importCSVFromUrl() {
  var url = promptUserForInput("Please enter the URL of the CSV file:");
  var contents = Utilities.parseCsv(UrlFetchApp.fetch(url));
  var sheetName = promptUserForInput("Please enter the sheet name for the imported data:");
  var importedSheetName = writeDataToSheet(contents, sheetName);
  displayToastAlert("The CSV file was successfully imported into " + importedSheetName + ".");
}

// Imports CSV data from Drive for multiple files
function importCSVFromDrive() {
  var fileNames = promptUserForInput("Please enter the names of the CSV files to import from Google Drive, separated by commas:");
  var fileNameArray = fileNames.split(',');
  
  for (var i = 0; i < fileNameArray.length; i++) {
    var currentFileName = fileNameArray[i].trim();
    var files = findFilesInDrive(currentFileName);
    
    if (files.length === 0) {
      displayToastAlert("No files with name \"" + currentFileName + "\" were found in Google Drive.");
    } else if (files.length > 1) {
      displayToastAlert("Multiple files with name " + currentFileName + " were found. This program does not support picking the right file yet.");
    } else {
      var file = files[0];
      var contents = Utilities.parseCsv(file.getBlob().getDataAsString());
      var importedSheetName = writeDataToSheet(contents, currentFileName);
      displayToastAlert("The CSV file \"" + currentFileName + "\" was successfully imported into sheet \"" + importedSheetName + "\".");
    }
  }
}

// Returns files in Google Drive that have a certain name
function findFilesInDrive(filename) {
  var files = DriveApp.getFilesByName(filename);
  var result = [];
  while (files.hasNext())
    result.push(files.next());
  return result;
}
