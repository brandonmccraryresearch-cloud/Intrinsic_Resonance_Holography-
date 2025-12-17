; ==============================================================================
; Intrinsic Resonance Holography (IRH) v21.1 - Windows Installer Script
;
; THEORETICAL FOUNDATION: Intrinsic_Resonance_Holography-v21.1.md §1.6
; Repository: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-
;
; This Inno Setup script creates a Windows installer (.exe) for IRH.
;
; To build the installer:
;   1. Install Inno Setup from https://jrsoftware.org/isinfo.php
;   2. Open this .iss file in Inno Setup Compiler
;   3. Click Build > Compile (or press Ctrl+F9)
;   4. The installer will be created in the Output directory
;
; ==============================================================================

#define MyAppName "Intrinsic Resonance Holography"
#define MyAppVersion "21.1.0"
#define MyAppPublisher "Brandon D. McCrary"
#define MyAppURL "https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-"
#define MyAppExeName "irh.bat"
#define MyAppAssocName MyAppName + " File"
#define MyAppAssocExt ".irh"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} v{#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\IRH
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
InfoBeforeFile=installer_readme.txt
OutputDir=Output
OutputBaseFilename=IRH-v{#MyAppVersion}-Windows-Setup
SetupIconFile=..\desktop\assets\irh_icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "Add IRH to system PATH"; GroupDescription: "Environment:"; Flags: checkedonce

[Files]
; Main repository files
Source: "..\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: ".git\*,*.pyc,__pycache__,*.egg-info,dist,build,*.exe,Output"

; Python embedded (optional - user can use their own Python)
; Source: "python-embed\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\IRH Documentation"; Filename: "{app}\README.md"
Name: "{group}\IRH Theory Manual"; Filename: "{app}\Intrinsic_Resonance_Holography-v21.1.md"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Registry]
; Add to PATH
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
; Post-install actions
Filename: "{app}\installers\install.bat"; Description: "Run post-installation setup"; Flags: nowait postinstall skipifsilent shellexec

[Code]
// Check if path needs to be added
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  // look for the path with leading and trailing semicolon
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;

// Custom wizard page for Python check
var
  PythonPage: TOutputMsgWizardPage;
  
procedure InitializeWizard;
begin
  PythonPage := CreateOutputMsgPage(wpWelcome,
    'Python Requirement',
    'IRH requires Python 3.10 or later',
    'IRH requires Python 3.10 or later to be installed on your system.' + #13#10 + #13#10 +
    'If you don''t have Python installed, please download it from:' + #13#10 +
    'https://www.python.org/downloads/' + #13#10 + #13#10 +
    'Make sure to check "Add Python to PATH" during installation.' + #13#10 + #13#10 +
    'Click Next to continue with the IRH installation.');
end;

// Check for Python at startup
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  
  // Check if Python is available
  if not Exec('python', '--version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if MsgBox('Python was not found in your PATH.' + #13#10 + #13#10 +
              'IRH requires Python 3.10 or later.' + #13#10 + #13#10 +
              'Would you like to continue anyway?' + #13#10 +
              '(You will need to install Python before using IRH)',
              mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
    end;
  end;
end;

[UninstallDelete]
; Clean up generated files
Type: filesandordirs; Name: "{app}\irh_venv"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: files; Name: "{app}\*.pyc"
