# python -m pip install Cython
# python -m pip install -r requirements.txt --force-reinstall

New-Item -ItemType Directory -Path .download -Force
Invoke-WebRequest -Uri https://github.com/Vogelwarte/SnowfinchWire.Common/archive/refs/tags/release-brood-analyzer.zip -OutFile .download/common.zip
Expand-Archive .download/common.zip -DestinationPath .download/
New-Item -ItemType Directory -Path sfw_brood/common -Force
Copy-Item -Recurse -Path .download/SnowfinchWire.Common-release-brood-analyzer/* -Destination sfw_brood/common/
Remove-Item -Recurse .download
