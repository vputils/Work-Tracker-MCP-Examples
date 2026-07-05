import * as os from 'os';

// Matches a tilde only if immediately followed by the end of string, "/" or "\".
const SAFE_HOME_REGEX = /^~(?=$|\/|\\)/;

/**
 * Expand the given path with the the user's home directory, 
 * in case it starts with the tilde (`~`), 
 * followed by "/", "\" or the end of the string.
 * @param path The path, which may contain a tilde (`~`) at the start, denoting the user's home directory.
 * @returns The home-expaned path, if containing a tilde (`~`) at the start, else the original path.
 */
export function expandHomeDir(path: string) {
    return path.replace(SAFE_HOME_REGEX, os.homedir());
}
