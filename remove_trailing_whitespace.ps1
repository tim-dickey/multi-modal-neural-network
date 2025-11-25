# Remove trailing whitespace from Python files
Get-ChildItem -Path src, tests -Recurse -Include "*.py" | Where-Object { $_.FullName -notmatch '\\\.venv\\' } | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    # Remove trailing whitespace from each line
    $lines = $content -split "`r?`n"
    $cleanedLines = $lines | ForEach-Object { $_.TrimEnd() }
    $cleanedContent = $cleanedLines -join "`n"
    # Only write if content changed
    if ($content -ne $cleanedContent) {
        Write-Host "Cleaning: $($_.FullName)"
        $cleanedContent | Set-Content $_.FullName -NoNewline
    }
}
Write-Host "Trailing whitespace removal complete."