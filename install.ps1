$python_path = $null
try
{
	$pversions = py -0p
	$version = $pversions.Where( {$_.Contains("3.9") }, 'First')
	if([bool] $version)
	{
		$python_path = $version.Split("  ")[-1].Trim()
	}
}
catch
{
	echo "py launcher not found"
}

if ($null -eq $python_path)
{

	echo "Python 3.9 not found, make sure it is installed and added to PATH"
	echo "ERROR"
	exit
}
echo "Found Python 3.9 at ${python_path}"

& $python_path --version

New-Item -ItemType Directory -Path .download -Force

# Common submodule
Invoke-WebRequest -Uri https://github.com/Vogelwarte/SnowfinchWire.Common/archive/refs/tags/release-brood-analyzer.zip -OutFile .download/common.zip
Expand-Archive .download/common.zip -DestinationPath .download/
New-Item -ItemType Directory -Path sfw_brood/common -Force
Copy-Item -Recurse -Path .download/SnowfinchWire.Common-release-brood-analyzer/* -Destination sfw_brood/common/

# BeggingCallsAnalyzer
Invoke-WebRequest -Uri https://github.com/Vogelwarte/SnowfinchWire.BeggingCallsAnalyzer/archive/refs/tags/1.1.zip -OutFile .download/sfw-bca.zip
Expand-Archive .download/sfw-bca.zip -DestinationPath .download/
New-Item -ItemType Directory -Path begging-analyzer -Force
Copy-Item -Recurse -Path .download/SnowfinchWire.BeggingCallsAnalyzer-1.1/* -Destination begging-analyzer/

# BeggingCallsAnalyzer common submodule
Invoke-WebRequest -Uri https://github.com/Vogelwarte/SnowfinchWire.Common/archive/refs/tags/v1.0-beggingcallsanalyzer.zip -OutFile .download/bca-common.zip
Expand-Archive .download/bca-common.zip -DestinationPath .download/
New-Item -ItemType Directory -Path begging-analyzer/beggingcallsanalyzer/common -Force
Copy-Item -Recurse -Path .download/SnowfinchWire.Common-1.0-beggingcallsanalyzer/* -Destination begging-analyzer/beggingcallsanalyzer/common/

Remove-Item -Recurse .download

& $python_path -m pip install --upgrade pip
& $python_path -m pip install -r begging-analyzer/requirements.txt
& $python_path -m pip install -r requirements.txt --force-reinstall
